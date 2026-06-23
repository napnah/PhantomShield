"""
mcu_attention_crypten.py
密文 Self-Attention（CrypTen 版本，作为完整系统的基线）
潘涵的 Π_softmax 完成后，替换 softmax 调用即可
"""
import torch
import crypten
import crypten.nn as cnn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from crypten_compat import patch_crypten_inprocess
patch_crypten_inprocess()
crypten.init_thread(0, 1)


def encrypt(x):
    """把张量加密成密文"""
    return crypten.cryptensor(x)


class MCUAttentionCrypTen:
    """
    密文单头 Self-Attention
    Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.scale = hidden_size ** 0.5

    def forward(self, X_enc, Wq, Wk, Wv):
        """
        X_enc: 加密的输入 (1, seq, hidden)  —— 用户私有
        Wq, Wk, Wv: 明文权重 (hidden, hidden) —— 这里先用明文，
                    实际中是P1的私有权重，也应加密
        返回：加密的注意力输出
        """
        # Q/K/V 投影（密文 × 明文权重）
        Q = X_enc.matmul(Wq)   # (1, seq, hidden)
        K = X_enc.matmul(Wk)
        V = X_enc.matmul(Wv)

        # 注意力分数 QK^T / sqrt(d)
        scores = Q.matmul(K.transpose(1, 2)) / self.scale  # (1, seq, seq)

        # Softmax（★ 这一行将来用潘涵的 Π_softmax 替换 ★）
        weights = scores.softmax(dim=-1)

        # 加权求和
        out = weights.matmul(V)  # (1, seq, hidden)
        return out


def test_attention():
    print('=== 密文 Self-Attention 测试 ===')

    torch.manual_seed(42)
    seq, hidden = 8, 32

    # 输入（用户私有数据）
    X = torch.randn(1, seq, hidden) * 0.1

    # 权重（服务商私有）
    Wq = torch.randn(hidden, hidden) * 0.1
    Wk = torch.randn(hidden, hidden) * 0.1
    Wv = torch.randn(hidden, hidden) * 0.1

    # ── 明文基准 ──
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = Q @ K.transpose(1, 2) / (hidden ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    expected = weights @ V
    print(f'明文输出 shape: {expected.shape}')

    # ── 密文推理 ──
    import time
    X_enc = encrypt(X)
    attn = MCUAttentionCrypTen(hidden)

    start = time.time()
    out_enc = attn.forward(X_enc, Wq, Wk, Wv)
    out = out_enc.get_plain_text()
    elapsed = time.time() - start

    error = (out - expected).abs().max().item()
    print(f'密文推理耗时: {elapsed:.2f}秒')
    print(f'最大误差: {error:.6f}')
    print(f'验证: {"通过" if error < 0.01 else "失败"}')

    print(f'\n明文输出（前2行2列）:\n{expected[0, :2, :2]}')
    print(f'密文输出（前2行2列）:\n{out[0, :2, :2]}')


if __name__ == '__main__':
    test_attention()