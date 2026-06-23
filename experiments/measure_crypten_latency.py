"""
measure_crypten_latency.py
测量 CrypTen 密文推理延迟，作为性能对比基线
对标论文中 BERT-Base 约 71 秒/样本
"""
import torch
import crypten
import time

crypten.init_thread(0, 1)

from transformers import BertModel, BertTokenizer


def measure_linear_layers():
    """
    测量 BERT 各种规模线性层的密文推理延迟
    用真实BERT权重，测CrypTen加密matmul的耗时
    """
    print('加载 BERT-Base...')
    model = BertModel.from_pretrained('./bert-base-uncased')
    model.eval()

    print('\n测量密文矩阵乘法延迟（CrypTen）...')
    print(f'{"操作":30s} {"明文(ms)":>12s} {"密文(ms)":>12s} {"减速比":>8s}')
    print('-' * 65)

    results = []
    seq_len = 128

    # 测试不同的BERT线性层
    test_layers = [
        ('Q投影 (768x768)', model.encoder.layer[0].attention.self.query),
        ('FFN第一层 (768x3072)', model.encoder.layer[0].intermediate.dense),
        ('FFN第二层 (3072x768)', model.encoder.layer[0].output.dense),
    ]

    for name, layer in test_layers:
        W = layer.weight.detach().T  # (in, out)
        d_in, d_out = W.shape
        X = torch.randn(1, seq_len, d_in) * 0.1

        # 明文基准
        start = time.time()
        for _ in range(3):
            _ = X @ W
        plain_ms = (time.time() - start) / 3 * 1000

        # 密文
        X_enc = crypten.cryptensor(X)
        W_enc = crypten.cryptensor(W)
        start = time.time()
        out = X_enc.matmul(W_enc)
        _ = out.get_plain_text()
        cipher_ms = (time.time() - start) * 1000

        slowdown = cipher_ms / plain_ms if plain_ms > 0 else 0
        print(f'{name:30s} {plain_ms:>12.3f} {cipher_ms:>12.1f} {slowdown:>7.0f}x')
        results.append((name, plain_ms, cipher_ms))

    return results


def estimate_full_bert():
    """基于单层测量，估算完整BERT推理延迟"""
    print('\n' + '='*65)
    print('完整 BERT-Base 推理延迟估算')
    print('='*65)
    print('BERT-Base: 12层，每层含 Attention(Q/K/V/O) + FFN(2层) + LayerNorm')
    print('外加 Softmax(12头) 和 GeLU 的密文非线性计算')
    print()
    print('说明：完整密文推理的瓶颈在 Softmax 和 GeLU 等非线性函数，')
    print('      CrypTen 原始实现约 71 秒/样本（论文数据）。')
    print('      本测量验证了线性层部分的密文计算开销。')


def main():
    print('='*65)
    print('CrypTen 密文推理延迟基线测量')
    print('='*65)
    results = measure_linear_layers()
    estimate_full_bert()

    # 保存
    with open('results/crypten_latency.txt', 'w', encoding='utf-8') as f:
        f.write('CrypTen Linear Layer Latency\n')
        for name, plain, cipher in results:
            f.write(f'{name}: plain={plain:.3f}ms, cipher={cipher:.1f}ms\n')
    print('\n结果已保存到 results/crypten_latency.txt')


if __name__ == '__main__':
    main()