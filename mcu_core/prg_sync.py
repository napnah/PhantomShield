"""
PRG 同步器
三方共享 seed_0，用 AES-CTR 生成相同的随机掩码序列
"""
import os
from Crypto.Cipher import AES
import struct

class PRGSync:
    def __init__(self, seed: bytes):
        """seed: 16字节的AES密钥"""
        assert len(seed) == 16, 'seed必须是16字节'
        self.seed = seed
        self.counter = 0

    def next(self, ring: int = 2**64) -> int:
        """生成下一个随机数，范围 [0, ring)"""
        nonce = struct.pack('>Q', self.counter) + b'\x00' * 8
        cipher = AES.new(self.seed, AES.MODE_CTR, nonce=nonce[:8],
                         initial_value=nonce[8:])
        rand_bytes = cipher.encrypt(b'\x00' * 8)
        self.counter += 1
        return int.from_bytes(rand_bytes, 'big') % ring

    def next_unit(self) -> float:
        """生成 [0, 1) 区间的浮点随机数（53 位精度）。

        超越函数协议（exp/softmax/gelu）在浮点域工作，
        需要实数掩码而非整数环元素，故提供此方法。
        三方使用相同种子时，next_unit() 序列也完全一致。
        """
        return self.next(2 ** 53) / float(2 ** 53)

    def next_real(self, high: float) -> float:
        """生成 [0, high) 区间的浮点随机数。"""
        return self.next_unit() * high

    @staticmethod
    def random_seed() -> bytes:
        """生成随机16字节种子"""
        return os.urandom(16)