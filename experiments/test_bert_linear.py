"""
test_bert_linear.py
用真实BERT权重测试密文线性层
- P0持有：一句话的embedding输出（用户私有）
- P1持有：BERT第一层attention的W_Q权重（服务商私有）
- 验证：密文矩阵乘法结果与明文一致
"""
import torch
import sys
import os
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BertModel, BertTokenizer
from mcu_core.prg_sync import PRGSync
from mcu_core.mock_comm import make_mock_comm
from transformer.mcu_linear import (
    MCULinearParty, MCULinearHP,
    share_tensor, reconstruct
)


def load_bert():
    print('加载BERT模型...')
    model = BertModel.from_pretrained('./bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    model.eval()
    print('加载完成')
    return model, tokenizer


def get_embedding(model, tokenizer, text: str):
    """获取输入文本的embedding（第一层attention的输入）"""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=16,       # 短序列，快速演示
        truncation=True,
        padding='max_length'
    )
    with torch.no_grad():
        # 只取embedding层输出，不跑attention
        embeddings = model.embeddings(
            inputs['input_ids'],
            token_type_ids=inputs['token_type_ids']
        )
    return embeddings  # shape: (1, 16, 768)


def get_wq(model):
    """取第一层attention的W_Q权重"""
    # BERT第0层，query的权重矩阵
    wq = model.encoder.layer[0].attention.self.query.weight
    return wq.detach().T  # shape: (768, 768)，转置使形状对齐


def main():
    model, tokenizer = load_bert()

    # P0持有的数据：用户输入
    text = "The patient has fever and cough."
    X = get_embedding(model, tokenizer, text)
    print(f'\nP0持有：输入embedding，shape = {X.shape}')

    # P1持有的数据：模型权重
    W = get_wq(model)
    print(f'P1持有：W_Q权重，shape = {W.shape}')

    # 明文结果（验证用）
    with torch.no_grad():
        expected = X @ W
    print(f'期望结果shape: {expected.shape}')

    # ── 为了演示速度，只取前4个token和前8维 ──
    X_small = X[:, :4, :8].contiguous()
    W_small = W[:8, :8].contiguous()
    expected_small = X_small @ W_small
    print(f'\n[演示用小矩阵] X: {X_small.shape}, W: {W_small.shape}')

    # 秘密共享
    print('\n开始秘密共享...')
    x0, x1, xs = share_tensor(X_small, PRGSync(bytes(range(48, 64))))
    w0, w1, ws = share_tensor(W_small, PRGSync(bytes(range(64, 80))))
    print(f'X已分成两份，P0持有份额0，P1持有份额1')
    print(f'W已分成两份，任何一方单独看到的都是随机数')

    # 初始化
    comm_p0, comm_p1, comm_hp = make_mock_comm()
    p0 = MCULinearParty(0, PRGSync(bytes(range(16))), comm_p0)
    p1 = MCULinearParty(1, PRGSync(bytes(range(16))), comm_p1)
    hp = MCULinearHP(PRGSync(bytes(range(16,32))),
                     PRGSync(bytes(range(32,48))), comm_hp)

    batch, seq, d_in = xs
    d_out = ws[1]
    mul_count = batch * seq * d_out * d_in
    print(f'\n需要 {mul_count} 次MPC乘法...')

    # 执行密文推理
    import time
    results = {}
    start = time.time()

    t0 = threading.Thread(
        target=lambda: results.__setitem__(
            'r0', p0.forward(x0, w0, xs, ws)))
    t1 = threading.Thread(
        target=lambda: results.__setitem__(
            'r1', p1.forward(x1, w1, xs, ws)))
    th = threading.Thread(target=lambda: hp.handle(mul_count))

    t0.start(); t1.start(); th.start()
    t0.join(); t1.join(); th.join()

    elapsed = time.time() - start
    print(f'密文推理完成，耗时: {elapsed:.2f}秒')

    # 合并结果
    mpc_result = reconstruct(results['r0'], results['r1'],
                             (batch, seq, d_out))

    error = (mpc_result - expected_small).abs().max().item()
    print(f'\n最大误差: {error:.8f}')
    print(f'验证: {"✓ 通过" if error < 0.01 else "✗ 失败"}')

    print('\n=== MPC结果（前2行）===')
    print(mpc_result[0, :2, :])
    print('\n=== 明文结果（前2行）===')
    print(expected_small[0, :2, :])


if __name__ == '__main__':
    main()