"""
benchmark_linear.py
测量 MCU 密文线性层在不同矩阵大小下的耗时
对比明文推理，生成报告里需要的实验数据
"""
import torch
import time
import threading
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcu_core.prg_sync import PRGSync
from mcu_core.mock_comm import make_mock_comm
from transformer.mcu_linear import (
    MCULinearParty, MCULinearHP,
    share_tensor, reconstruct
)


def run_mpc_linear(batch, seq, d_in, d_out, repeat=3):
    """
    测量一次MPC线性层的平均耗时
    返回：(平均耗时秒, 最大误差)
    """
    X = torch.randn(batch, seq, d_in) * 0.1
    W = torch.randn(d_in, d_out) * 0.1
    expected = X @ W

    times = []
    last_error = 0

    for _ in range(repeat):
        x0, x1, xs = share_tensor(X, PRGSync(bytes(range(48, 64))))
        w0, w1, ws = share_tensor(W, PRGSync(bytes(range(64, 80))))

        comm_p0, comm_p1, comm_hp = make_mock_comm()
        p0 = MCULinearParty(0, PRGSync(bytes(range(16))), comm_p0)
        p1 = MCULinearParty(1, PRGSync(bytes(range(16))), comm_p1)
        hp = MCULinearHP(PRGSync(bytes(range(16, 32))),
                         PRGSync(bytes(range(32, 48))), comm_hp)

        mul_count = batch * seq * d_out * d_in
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
        times.append(elapsed)

        mpc_result = reconstruct(results['r0'], results['r1'],
                                 (batch, seq, d_out))
        last_error = (mpc_result - expected).abs().max().item()

    return sum(times) / len(times), last_error


def run_plaintext_linear(batch, seq, d_in, d_out, repeat=3):
    """测量明文线性层耗时"""
    X = torch.randn(batch, seq, d_in) * 0.1
    W = torch.randn(d_in, d_out) * 0.1

    times = []
    for _ in range(repeat):
        start = time.time()
        _ = X @ W
        times.append(time.time() - start)

    return sum(times) / len(times)


def main():
    print('=' * 65)
    print('MCU 密文线性层性能基准测试')
    print('=' * 65)
    print(f'{"矩阵规模":20s} {"明文(ms)":>10s} {"MPC(ms)":>10s} '
          f'{"减速比":>8s} {"误差":>12s}')
    print('-' * 65)

    # 测试不同规模（从小到大）
    configs = [
        # (batch, seq, d_in, d_out, 描述)
        (1,  4,   8,   8,  '极小'),
        (1,  4,  32,  32,  '小'),
        (1,  4,  64,  64,  '中小'),
        (1,  4, 128, 128,  '中'),
        (1,  8, 128, 128,  '中（长序列）'),
        (1,  4, 256, 256,  '较大'),
        (1,  4, 512, 512,  'BERT隐藏层维度'),
    ]

    results = []
    for batch, seq, d_in, d_out, desc in configs:
        mul_count = batch * seq * d_in * d_out
        label = f'({batch},{seq},{d_in},{d_out})'

        # 规模太大跳过（避免太慢）
        if mul_count > 50000:
            print(f'{label:20s} {"--":>10s} {"太慢跳过":>10s}')
            continue

        plain_t  = run_plaintext_linear(batch, seq, d_in, d_out)
        mpc_t, err = run_mpc_linear(batch, seq, d_in, d_out, repeat=2)

        slowdown = mpc_t / plain_t if plain_t > 0 else float('inf')

        print(f'{label:20s} {plain_t*1000:>10.3f} {mpc_t*1000:>10.1f} '
              f'{slowdown:>8.0f}x {err:>12.2e}')

        results.append({
            'desc': desc,
            'batch': batch, 'seq': seq,
            'd_in': d_in, 'd_out': d_out,
            'mul_count': mul_count,
            'plain_ms': plain_t * 1000,
            'mpc_ms':   mpc_t   * 1000,
            'slowdown': slowdown,
            'error':    err
        })

    print('=' * 65)

    # 输出报告里用的关键数据
    print('\n关键结论：')
    if results:
        avg_slowdown = sum(r['slowdown'] for r in results) / len(results)
        max_error    = max(r['error'] for r in results)
        print(f'  平均减速比: {avg_slowdown:.0f}x（相比明文推理）')
        print(f'  最大误差:   {max_error:.2e}（定点数精度损失）')
        print(f'  乘法次数:   每次MPC线性层需要 d_in × d_out 次协议调用')

    print('\nBERT 推理规模预估：')
    print('  BERT-Base 单层 Attention Q 投影: (1, 512, 768, 768)')
    print(f'  需要约 {512*768*768:,} 次 MPC 乘法')
    print(f'  按当前速度预估约 {512*768*768/256*0.05/60:.1f} 分钟/层')

    # 保存结果到CSV
    import csv
    csv_path = 'experiments/benchmark_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'\n结果已保存到 {csv_path}')


if __name__ == '__main__':
    main()