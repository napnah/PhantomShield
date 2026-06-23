"""
party.py —— 三方进程启动入口

用法：
    # 三个独立终端分别运行：
    python party.py --role hp
    python party.py --role p0 --input "患者病历文本"
    python party.py --role p1 --model ./bert-base-uncased
"""
import argparse
import socket
import threading
import time
from mcu_core.comm import send_msg, recv_msg, HOST, PORT_HP, PORT_P0, PORT_P1
from mcu_core.prg_sync import PRGSync

# ── 固定种子（演示用，生产环境应用 DH 密钥协商）──
SHARED_SEED  = bytes(range(16))
SEED_HP_P0   = bytes(range(16, 32))
SEED_HP_P1   = bytes(range(32, 48))


class HPProcess:
    """辅助方进程"""

    def __init__(self):
        self.prg_p0 = PRGSync(SEED_HP_P0)
        self.prg_p1 = PRGSync(SEED_HP_P1)
        self.conn_p0 = None
        self.conn_p1 = None

    def start(self):
        print('[HP] 启动，等待 P0 和 P1 连接...')
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT_HP))
        server.listen(2)

        # 接受两个连接
        connections = {}
        while len(connections) < 2:
            conn, addr = server.accept()
            msg = recv_msg(conn)
            role = msg['role']
            connections[role] = conn
            print(f'[HP] {role} 已连接')

        self.conn_p0 = connections['p0']
        self.conn_p1 = connections['p1']
        print('[HP] 所有参与方已就绪，开始协议')
        self._run()

    def _run(self):
        """主循环：处理来自两方的协议请求"""
        while True:
            try:
                msg0 = recv_msg(self.conn_p0)
                msg1 = recv_msg(self.conn_p1)
                op = msg0.get('op')

                if op == 'mul':
                    self._handle_mul(msg0, msg1)
                elif op == 'done':
                    print('[HP] 协议完成')
                    break
                else:
                    print(f'[HP] 未知操作: {op}')
            except Exception as e:
                print(f'[HP] 错误: {e}')
                break

    def _handle_mul(self, msg0, msg1):
        L = 2**64
        mx = (msg0['mx'] + msg1['mx']) % L
        my = (msg0['my'] + msg1['my']) % L
        product = (mx * my) % L
        s0 = self.prg_p0.next(L)
        s1 = (product - s0) % L
        send_msg(self.conn_p0, {'share': s0})
        send_msg(self.conn_p1, {'share': s1})
        print(f'[HP] 处理乘法请求完成')


class P0Process:
    """用户方进程"""

    def __init__(self, input_text: str):
        self.input   = input_text
        self.prg0    = PRGSync(SHARED_SEED)
        self.conn_hp = None

    def start(self):
        print('[P0] 连接到 HP...')
        self.conn_hp = self._connect(PORT_HP, 'p0')
        print('[P0] 已连接到 HP')
        self._run()

    def _connect(self, port, role):
        while True:
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.connect((HOST, port))
                send_msg(conn, {'role': role})
                return conn
            except ConnectionRefusedError:
                time.sleep(0.5)

    def _run(self):
        """演示：用MPC计算两个数的乘积"""
        L = 2**64

        # 秘密值（演示用固定值，后续换成真实BERT输入）
        x, y = 12345, 67890
        x0 = 999999
        x1 = (x - x0) % L
        y0 = 888888
        y1 = (y - y0) % L

        print(f'[P0] 持有份额: x0={x0}, y0={y0}')
        print(f'[P0] 真实值（仅P0知道）: x={x}, y={y}')

        # 执行乘法协议
        r_x = self.prg0.next(L)
        r_y = self.prg0.next(L)

        send_msg(self.conn_hp, {
            'op': 'mul',
            'mx': (x0 + r_x) % L,
            'my': (y0 + r_y) % L
        })

        msg = recv_msg(self.conn_hp)
        s0 = msg['share']

        correction = (x0*r_y + y0*r_x + r_x*r_y) % L
        result0 = (s0 - correction) % L

        print(f'[P0] 本方结果份额: {result0}')

        # 通知完成
        send_msg(self.conn_hp, {'op': 'done'})


class P1Process:
    """服务商进程"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.prg0       = PRGSync(SHARED_SEED)
        self.conn_hp    = None

    def start(self):
        print('[P1] 连接到 HP...')
        self.conn_hp = self._connect(PORT_HP, 'p1')
        print('[P1] 已连接到 HP')
        self._run()

    def _connect(self, port, role):
        while True:
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.connect((HOST, port))
                send_msg(conn, {'role': role})
                return conn
            except ConnectionRefusedError:
                time.sleep(0.5)

    def _run(self):
        L = 2**64

        x1 = (12345 - 999999) % L
        y1 = (67890 - 888888) % L

        print(f'[P1] 持有份额: x1={x1}, y1={y1}')
        print(f'[P1] 模型路径: {self.model_path}（后续接入BERT）')

        r_x = self.prg0.next(L)
        r_y = self.prg0.next(L)

        send_msg(self.conn_hp, {
            'op': 'mul',
            'mx': x1 % L,
            'my': y1 % L
        })

        msg    = recv_msg(self.conn_hp)
        s1     = msg['share']
        correction = (x1*r_y + y1*r_x) % L
        result1    = (s1 - correction) % L

        print(f'[P1] 本方结果份额: {result1}')

        # P1主动把份额发给P0合并（演示用）
        print(f'[P1] 发送份额给P0用于合并结果...')
        print(f'[P1] result1 = {result1}')
        send_msg(self.conn_hp, {'op': 'done'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role',  required=True, choices=['hp','p0','p1'])
    parser.add_argument('--input', default='患者病历：发烧38.5度，咳嗽三天')
    parser.add_argument('--model', default='./bert-base-uncased')
    args = parser.parse_args()

    if args.role == 'hp':
        HPProcess().start()
    elif args.role == 'p0':
        P0Process(args.input).start()
    elif args.role == 'p1':
        P1Process(args.model).start()