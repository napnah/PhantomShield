"""
三方通信层
P0（用户方）、P1（服务商）、HP（辅助方）通过 socket 通信
"""
import socket
import json
import struct

PORT_HP = 9000
PORT_P0 = 9001
PORT_P1 = 9002
HOST = '127.0.0.1'

def send_msg(sock, data: dict):
    """发送消息：先发4字节长度，再发JSON内容"""
    msg = json.dumps(data).encode('utf-8')
    length = struct.pack('>I', len(msg))
    sock.sendall(length + msg)

def recv_msg(sock) -> dict:
    """接收消息：先读4字节长度，再读内容"""
    raw_len = _recv_exact(sock, 4)
    length = struct.unpack('>I', raw_len)[0]
    raw_data = _recv_exact(sock, length)
    return json.loads(raw_data.decode('utf-8'))

def _recv_exact(sock, n: int) -> bytes:
    """精确接收n个字节"""
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError('连接断开')
        data += chunk
    return data