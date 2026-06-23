"""
mock_comm.py —— 本地单文件协议测试接口
使用线程安全的 Queue，支持多线程并发测试
"""
import queue


class MockCommParty:
    """P0 或 P1 用的假通信对象"""

    def __init__(self, my_inbox: queue.Queue, hp_inbox: queue.Queue):
        self._my = my_inbox
        self._hp = hp_inbox

    def send_to_hp(self, msg: dict):
        self._hp.put(msg)

    def recv_from_hp(self) -> dict:
        return self._my.get()  # 阻塞等待，直到HP放入消息


class MockCommHP:
    """HP 用的假通信对象"""

    def __init__(self,
                 from_p0: queue.Queue, from_p1: queue.Queue,
                 to_p0:   queue.Queue, to_p1:   queue.Queue):
        self._from_p0 = from_p0
        self._from_p1 = from_p1
        self._to_p0   = to_p0
        self._to_p1   = to_p1

    def recv_from_p0(self) -> dict:
        return self._from_p0.get()

    def recv_from_p1(self) -> dict:
        return self._from_p1.get()

    def send_to_p0(self, msg: dict):
        self._to_p0.put(msg)

    def send_to_p1(self, msg: dict):
        self._to_p1.put(msg)


def make_mock_comm():
    """
    创建三方通信对象

    返回：(comm_p0, comm_p1, comm_hp)
    """
    hp_inbox_p0 = queue.Queue()  # P0 → HP
    hp_inbox_p1 = queue.Queue()  # P1 → HP
    p0_inbox    = queue.Queue()  # HP → P0
    p1_inbox    = queue.Queue()  # HP → P1

    comm_p0 = MockCommParty(my_inbox=p0_inbox,    hp_inbox=hp_inbox_p0)
    comm_p1 = MockCommParty(my_inbox=p1_inbox,    hp_inbox=hp_inbox_p1)
    comm_hp = MockCommHP(from_p0=hp_inbox_p0, from_p1=hp_inbox_p1,
                         to_p0=p0_inbox,      to_p1=p1_inbox)

    return comm_p0, comm_p1, comm_hp