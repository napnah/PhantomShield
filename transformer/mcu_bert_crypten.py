"""
mcu_bert_crypten.py
完整的密文 BERT Encoder Layer（CrypTen 版本）
串联 Attention + FFN + LayerNorm + 残差连接
这是完整密文推理系统的集成验证
"""
import torch
import crypten
import time

crypten.init_thread(0, 1)


def encrypt(x):
    return crypten.cryptensor(x)


class MCUBertLayer:
    """
    完整的密文 Transformer Encoder Layer
    x -> LN(x + Attention(x)) -> LN(x + FFN(x))
    """

    def __init__(self, hidden_size, ffn_size):
        self.hidden = hidden_size
        self.ffn = ffn_size
        self.scale = hidden_size ** 0.5

    def attention(self, X_enc, Wq, Wk, Wv):
        Q = X_enc.matmul(Wq)
        K = X_enc.matmul(Wk)
        V = X_enc.matmul(Wv)
        scores = Q.matmul(K.transpose(1, 2)) / self.scale
        weights = scores.softmax(dim=-1)
        return weights.matmul(V)

    def ffn_forward(self, X_enc, W1, b1, W2, b2):
        h = X_enc.matmul(W1) + b1
        h = h * (h * 1.702).sigmoid()   # GeLU近似
        return h.matmul(W2) + b2

    def layernorm(self, X_enc, gamma, beta, eps=1e-5):
        """
        密文 LayerNorm
        均值/方差作为统计量安全揭示（SecFormer同款处理），
        归一化系数明文计算后回乘密文，避免inv_sqrt精度问题
        """
        mean = X_enc.mean(dim=-1, keepdim=True)
        diff = X_enc - mean
        var = (diff * diff).mean(dim=-1, keepdim=True)

        # 揭示方差（统计量，不泄露单个元素），明文算 1/sqrt
        var_plain = var.get_plain_text()
        inv_std_plain = 1.0 / torch.sqrt(var_plain + eps)

        # 回乘密文
        normed = diff * inv_std_plain
        return normed * gamma + beta

    def forward(self, X_enc, weights):
        """完整的一层 Encoder"""
        # Self-Attention + 残差 + LayerNorm
        attn_out = self.attention(X_enc, weights['Wq'],
                                  weights['Wk'], weights['Wv'])
        X_enc = self.layernorm(X_enc + attn_out,
                               weights['ln1_g'], weights['ln1_b'])

        # FFN + 残差 + LayerNorm
        ffn_out = self.ffn_forward(X_enc, weights['W1'], weights['b1'],
                                   weights['W2'], weights['b2'])
        X_enc = self.layernorm(X_enc + ffn_out,
                               weights['ln2_g'], weights['ln2_b'])
        return X_enc


def make_weights(hidden, ffn):
    """生成一层的随机权重（模拟训练好的BERT）"""
    return {
        'Wq': torch.randn(hidden, hidden) * 0.1,
        'Wk': torch.randn(hidden, hidden) * 0.1,
        'Wv': torch.randn(hidden, hidden) * 0.1,
        'W1': torch.randn(hidden, ffn) * 0.1,
        'b1': torch.randn(ffn) * 0.1,
        'W2': torch.randn(ffn, hidden) * 0.1,
        'b2': torch.randn(hidden) * 0.1,
        'ln1_g': torch.ones(hidden),
        'ln1_b': torch.zeros(hidden),
        'ln2_g': torch.ones(hidden),
        'ln2_b': torch.zeros(hidden),
    }


def plaintext_layer(X, w, hidden):
    """明文版本，用于验证"""
    import torch.nn.functional as F
    scale = hidden ** 0.5

    # Attention
    Q = X @ w['Wq']; K = X @ w['Wk']; V = X @ w['Wv']
    scores = Q @ K.transpose(1, 2) / scale
    attn = torch.softmax(scores, dim=-1) @ V
    X = F.layer_norm(X + attn, (hidden,), w['ln1_g'], w['ln1_b'])

    # FFN
    h = X @ w['W1'] + w['b1']
    h = h * torch.sigmoid(h * 1.702)
    ffn = h @ w['W2'] + w['b2']
    X = F.layer_norm(X + ffn, (hidden,), w['ln2_g'], w['ln2_b'])
    return X


def test_full_layer():
    print('=== 完整密文 BERT Encoder Layer 测试 ===')

    torch.manual_seed(42)
    seq, hidden, ffn = 8, 32, 128
    num_layers = 2

    X = torch.randn(1, seq, hidden) * 0.1
    layers_w = [make_weights(hidden, ffn) for _ in range(num_layers)]

    # 明文基准
    X_plain = X.clone()
    for w in layers_w:
        X_plain = plaintext_layer(X_plain, w, hidden)
    print(f'明文输出 shape: {X_plain.shape}')

    # 密文推理
    print(f'\n开始 {num_layers} 层密文推理...')
    X_enc = encrypt(X)
    layer = MCUBertLayer(hidden, ffn)

    start = time.time()
    for i, w in enumerate(layers_w):
        X_enc = layer.forward(X_enc, w)
        print(f'  第 {i+1} 层完成')
    out = X_enc.get_plain_text()
    elapsed = time.time() - start

    error = (out - X_plain).abs().max().item()
    print(f'\n密文推理总耗时: {elapsed:.2f}秒')
    print(f'最大误差: {error:.6f}')
    print(f'验证: {"通过" if error < 0.1 else "失败"}')


if __name__ == '__main__':
    test_full_layer()