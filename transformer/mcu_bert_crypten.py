"""
mcu_bert_crypten.py
瀹屾暣鐨勫瘑鏂?BERT Encoder Layer锛圕rypTen 鐗堟湰锛?
涓茶仈 Attention + FFN + LayerNorm + 娈嬪樊杩炴帴
杩欐槸瀹屾暣瀵嗘枃鎺ㄧ悊绯荤粺鐨勯泦鎴愰獙璇?
"""
import torch
import crypten
import time
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from crypten_compat import patch_crypten_inprocess

patch_crypten_inprocess()
crypten.init_thread(0, 1)


def encrypt(x):
    return crypten.cryptensor(x)


class MCUBertLayer:
    """
    瀹屾暣鐨勫瘑鏂?Transformer Encoder Layer
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
        h = h * (h * 1.702).sigmoid()   # GeLU杩戜技
        return h.matmul(W2) + b2

    def layernorm(self, X_enc, gamma, beta, eps=1e-5):
        """
        瀵嗘枃 LayerNorm
        鍧囧€?鏂瑰樊浣滀负缁熻閲忓畨鍏ㄦ彮绀猴紙SecFormer鍚屾澶勭悊锛夛紝
        褰掍竴鍖栫郴鏁版槑鏂囪绠楀悗鍥炰箻瀵嗘枃锛岄伩鍏峣nv_sqrt绮惧害闂
        """
        mean = X_enc.mean(dim=-1, keepdim=True)
        diff = X_enc - mean
        var = (diff * diff).mean(dim=-1, keepdim=True)

        # 鎻ず鏂瑰樊锛堢粺璁￠噺锛屼笉娉勯湶鍗曚釜鍏冪礌锛夛紝鏄庢枃绠?1/sqrt
        var_plain = var.get_plain_text()
        inv_std_plain = 1.0 / torch.sqrt(var_plain + eps)

        # 鍥炰箻瀵嗘枃
        normed = diff * inv_std_plain
        return normed * gamma + beta

    def forward(self, X_enc, weights):
        """瀹屾暣鐨勪竴灞?Encoder"""
        # Self-Attention + 娈嬪樊 + LayerNorm
        attn_out = self.attention(X_enc, weights['Wq'],
                                  weights['Wk'], weights['Wv'])
        X_enc = self.layernorm(X_enc + attn_out,
                               weights['ln1_g'], weights['ln1_b'])

        # FFN + 娈嬪樊 + LayerNorm
        ffn_out = self.ffn_forward(X_enc, weights['W1'], weights['b1'],
                                   weights['W2'], weights['b2'])
        X_enc = self.layernorm(X_enc + ffn_out,
                               weights['ln2_g'], weights['ln2_b'])
        return X_enc


def make_weights(hidden, ffn):
    """鐢熸垚涓€灞傜殑闅忔満鏉冮噸锛堟ā鎷熻缁冨ソ鐨凚ERT锛?""
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
    """鏄庢枃鐗堟湰锛岀敤浜庨獙璇?""
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
    print('=== 瀹屾暣瀵嗘枃 BERT Encoder Layer 娴嬭瘯 ===')

    torch.manual_seed(42)
    seq, hidden, ffn = 8, 32, 128
    num_layers = 2

    X = torch.randn(1, seq, hidden) * 0.1
    layers_w = [make_weights(hidden, ffn) for _ in range(num_layers)]

    # 鏄庢枃鍩哄噯
    X_plain = X.clone()
    for w in layers_w:
        X_plain = plaintext_layer(X_plain, w, hidden)
    print(f'鏄庢枃杈撳嚭 shape: {X_plain.shape}')

    # 瀵嗘枃鎺ㄧ悊
    print(f'\n寮€濮?{num_layers} 灞傚瘑鏂囨帹鐞?..')
    X_enc = encrypt(X)
    layer = MCUBertLayer(hidden, ffn)

    start = time.time()
    for i, w in enumerate(layers_w):
        X_enc = layer.forward(X_enc, w)
        print(f'  绗?{i+1} 灞傚畬鎴?)
    out = X_enc.get_plain_text()
    elapsed = time.time() - start

    error = (out - X_plain).abs().max().item()
    print(f'\n瀵嗘枃鎺ㄧ悊鎬昏€楁椂: {elapsed:.2f}绉?)
    print(f'鏈€澶ц宸? {error:.6f}')
    print(f'楠岃瘉: {"閫氳繃" if error < 0.1 else "澶辫触"}')


if __name__ == '__main__':
    test_full_layer()
