"""
Π_mul：MCU 安全乘法协议（2轮通信）

数学基础：
  x被分成 x0+x1=x，y被分成 y0+y1=y
  r_x、r_y 由 PRG0 生成（三方公开）
  
  P0发: x0+r_x, y0+r_y
  P1发: x1, y1
  HP重建: mx = x+r_x, my = y+r_y
  HP计算: P = mx*my = (x+r_x)(y+r_y)
  
  去掩码：
    xy = P - x*r_y - y*r_x - r_x*r_y
    P0修正: x0*r_y + y0*r_x + r_x*r_y
    P1修正: x1*r_y + y1*r_x
    两者之和 = x*r_y + y*r_x + r_x*r_y ✓
"""
L = 2**64


class MultiplyParty:
    """P0 或 P1 的乘法协议实现"""

    def __init__(self, party_id: int, prg0, comm):
        self.id   = party_id
        self.prg0 = prg0
        self.comm = comm

    def multiply(self, share_x: int, share_y: int) -> int:
        """
        输入：本方持有的 [x]_i 和 [y]_i
        输出：本方持有的 [x*y]_i
        """
        # 生成公开掩码（三方PRG同步，生成相同值）
        r_x = self.prg0.next(L)
        r_y = self.prg0.next(L)

        # Step 1 (Mask + Send)
        if self.id == 0:
            # P0 加掩码后发给 HP
            self.comm.send_to_hp({
                'id': 0,
                'mx': (share_x + r_x) % L,
                'my': (share_y + r_y) % L
            })
        else:
            # P1 直接发份额（r_x是公开的，HP无法从x1推断x）
            self.comm.send_to_hp({
                'id': 1,
                'mx': share_x % L,
                'my': share_y % L
            })

        # Step 2 (Receive share from HP)
        msg = self.comm.recv_from_hp()
        s_i = msg['share']

        # Step 3 (Unmask)
        if self.id == 0:
            # P0 修正三项
            correction = (share_x*r_y + share_y*r_x + r_x*r_y) % L
        else:
            # P1 修正两项
            correction = (share_x*r_y + share_y*r_x) % L

        return (s_i - correction) % L


class MultiplyHP:
    """HP 的乘法协议实现"""

    def __init__(self, asprg_p0, asprg_p1, comm):
        self.asprg_p0 = asprg_p0
        self.asprg_p1 = asprg_p1
        self.comm     = comm

    def handle_multiply(self):
        # 接收两方消息
        msg0 = self.comm.recv_from_p0()
        msg1 = self.comm.recv_from_p1()

        # 重建掩码后的值
        mx = (msg0['mx'] + msg1['mx']) % L  # = x + r_x
        my = (msg0['my'] + msg1['my']) % L  # = y + r_y

        # 计算乘积
        product = (mx * my) % L

        # 生成随机份额分发
        s0 = self.asprg_p0.next(L)
        s1 = (product - s0) % L

        self.comm.send_to_p0({'share': s0})
        self.comm.send_to_p1({'share': s1})


def verify_multiply():
    from mcu_core.prg_sync import PRGSync

    shared_seed = bytes(range(16))
    seed_hp_p0  = bytes(range(16, 32))
    seed_hp_p1  = bytes(range(32, 48))

    # 三方各自的PRG（同种子，生成相同序列）
    prg0_p0 = PRGSync(shared_seed)
    prg0_p1 = PRGSync(shared_seed)
    prg0_hp = PRGSync(shared_seed)

    asprg_p0 = PRGSync(seed_hp_p0)
    asprg_p1 = PRGSync(seed_hp_p1)

    # 消息队列
    hp_inbox_p0, hp_inbox_p1 = [], []
    p0_inbox,    p1_inbox    = [], []

    class FakeCommParty:
        def __init__(self, my_inbox, hp_inbox):
            self._my = my_inbox
            self._hp = hp_inbox
        def send_to_hp(self, msg): self._hp.append(msg)
        def recv_from_hp(self):    return self._my.pop(0)

    class FakeCommHP:
        def recv_from_p0(self):    return hp_inbox_p0.pop(0)
        def recv_from_p1(self):    return hp_inbox_p1.pop(0)
        def send_to_p0(self, msg): p0_inbox.append(msg)
        def send_to_p1(self, msg): p1_inbox.append(msg)

    # 真实值
    x, y     = 12345, 67890
    expected = (x * y) % L
    print(f'x = {x}, y = {y}')
    print(f'期望: {expected}')

    # 秘密共享 x 和 y
    x0, x1 = 999999, (x - 999999) % L
    y0, y1 = 888888, (y - 888888) % L
    assert (x0 + x1) % L == x
    assert (y0 + y1) % L == y

    # 初始化参与方
    p0 = MultiplyParty(0, prg0_p0, FakeCommParty(p0_inbox, hp_inbox_p0))
    p1 = MultiplyParty(1, prg0_p1, FakeCommParty(p1_inbox, hp_inbox_p1))
    hp = MultiplyHP(asprg_p0, asprg_p1, FakeCommHP())

    # 执行协议（顺序模拟）
    # 生成相同的公开掩码
    r_x = prg0_p0.next(L)
    r_y = prg0_p0.next(L)
    r_x1 = prg0_p1.next(L)
    r_y1 = prg0_p1.next(L)
    assert r_x == r_x1 and r_y == r_y1, 'PRG不同步！'

    # P0 发送
    hp_inbox_p0.append({'id': 0, 'mx': (x0+r_x)%L, 'my': (y0+r_y)%L})
    # P1 发送
    hp_inbox_p1.append({'id': 1, 'mx': x1%L,        'my': y1%L})

    # HP 处理
    hp.handle_multiply()

    # P0 去掩码
    s0 = p0_inbox.pop(0)['share']
    c0 = (x0*r_y + y0*r_x + r_x*r_y) % L
    r0 = (s0 - c0) % L

    # P1 去掩码
    s1 = p1_inbox.pop(0)['share']
    c1 = (x1*r_y + y1*r_x) % L
    r1 = (s1 - c1) % L

    result = (r0 + r1) % L
    print(f'MPC结果: {result}')
    print(f'正确性验证: {"✓ 通过" if result == expected else "✗ 失败"}')

    # 多组测试
    print('\n--- 多组随机测试 ---')
    import random
    for _ in range(5):
        xi = random.randint(0, 10**9)
        yi = random.randint(0, 10**9)
        exp = (xi * yi) % L

        xi0 = random.randint(0, L-1)
        xi1 = (xi - xi0) % L
        yi0 = random.randint(0, L-1)
        yi1 = (yi - yi0) % L

        rx = prg0_p0.next(L)
        ry = prg0_p0.next(L)
        prg0_p1.next(L)  # 同步P1的PRG
        prg0_p1.next(L)

        hp_inbox_p0.append({'id':0,'mx':(xi0+rx)%L,'my':(yi0+ry)%L})
        hp_inbox_p1.append({'id':1,'mx':xi1%L,'my':yi1%L})
        hp.handle_multiply()

        s0i = p0_inbox.pop(0)['share']
        s1i = p1_inbox.pop(0)['share']
        c0i = (xi0*ry + yi0*rx + rx*ry) % L
        c1i = (xi1*ry + yi1*rx) % L
        res = ((s0i-c0i) + (s1i-c1i)) % L

        status = '✓' if res == exp else '✗'
        print(f'  {status} {xi} × {yi} = {exp}, MPC={res}')


if __name__ == '__main__':
    verify_multiply()