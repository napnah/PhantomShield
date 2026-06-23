"""
mcu_ffn_crypten.py
密文 FFN（CrypTen 版本）
FFN(x) = GeLU(x W1 + b1) W2 + b2
潘涵的 Π_gelu 完成后替换 gelu 调用
"""
import torch
import crypten
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from crypten_compat import patch_crypten_inprocess
patch_crypten_inprocess()
crypten.init_thread(0, 1)


def encrypt(x):
    return crypten.cryptensor(x)


class MCUFFNCrypTen:
    """密文前馈网络"""

    def __init__(self, hidden_size, ffn_size):
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size

    def forward(self, X_enc, W1, b1, W2, b2):
        """
        X_enc: 加密输入 (1, seq, hidden)
        W1: (hidden, ffn), b1: (ffn,)
        W2: (ffn, hidden), b2: (hidden,)
        """
        # 第一层：扩展维度
        h = X_enc.matmul(W1) + b1          # (1, seq, ffn)

        # GeLU 的 sigmoid 近似：GeLU(x) ≈ x · sigmoid(1.702x)
        # （★ 将来用潘涵的 Π_gelu 替换，公式一致 ★）
        h = h * (h * 1.702).sigmoid()

        # 第二层：压缩回原维度
        out = h.matmul(W2) + b2            # (1, seq, hidden)
        return out


def test_ffn():
    print('=== 密文 FFN 测试 ===')

    torch.manual_seed(42)
    seq, hidden = 8, 32
    ffn = hidden * 4

    X = torch.randn(1, seq, hidden) * 0.1
    W1 = torch.randn(hidden, ffn) * 0.1
    b1 = torch.randn(ffn) * 0.1
    W2 = torch.randn(ffn, hidden) * 0.1
    b2 = torch.randn(hidden) * 0.1

    # 明文基准
    h = X @ W1 + b1
    h = torch.nn.functional.gelu(h)
    expected = h @ W2 + b2
    print(f'明文输出 shape: {expected.shape}')

    # 密文推理
    import time
    X_enc = encrypt(X)
    ffn_layer = MCUFFNCrypTen(hidden, ffn)

    start = time.time()
    out_enc = ffn_layer.forward(X_enc, W1, b1, W2, b2)
    out = out_enc.get_plain_text()
    elapsed = time.time() - start

    error = (out - expected).abs().max().item()
    print(f'密文推理耗时: {elapsed:.2f}秒')
    print(f'最大误差: {error:.6f}')
    print(f'验证: {"通过" if error < 0.05 else "失败"}')


if __name__ == '__main__':
    test_ffn()