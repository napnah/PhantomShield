//! criterion 基准：批量协议吞吐。

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use mcu_rust::simulate;

fn seeds() -> ([u8; 16], [u8; 16]) {
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    for i in 0..16 {
        a[i] = i as u8;
        b[i] = (i + 16) as u8;
    }
    (a, b)
}

fn make_shares(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x0: Vec<f64> = (0..n).map(|i| ((i % 21) as f64) - 10.0).collect();
    let x1: Vec<f64> = (0..n).map(|i| 0.01 * (i as f64)).collect();
    (x0, x1)
}

fn bench_exp(c: &mut Criterion) {
    let (ss, sh) = seeds();
    let n = 512 * 512;
    let (x0, x1) = make_shares(n);
    c.bench_function("exp_batch_512x512", |b| {
        b.iter_batched(
            || (),
            |_| simulate::exp_batch(&x0, &x1, &ss, &sh),
            BatchSize::SmallInput,
        )
    });
}

fn bench_gelu(c: &mut Criterion) {
    let (ss, sh) = seeds();
    let n = 512 * 512;
    let (x0, x1) = make_shares(n);
    c.bench_function("gelu_batch_512x512", |b| {
        b.iter_batched(
            || (),
            |_| simulate::gelu_batch(&x0, &x1, &ss, &sh),
            BatchSize::SmallInput,
        )
    });
}

fn bench_softmax(c: &mut Criterion) {
    let (ss, sh) = seeds();
    let n = 4096;
    let k = 64;
    let (x0, x1) = make_shares(n * k);
    c.bench_function("softmax_batch_4096x64", |b| {
        b.iter_batched(
            || (),
            |_| simulate::softmax_batch(&x0, &x1, n, k, &ss, &sh),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_exp, bench_gelu, bench_softmax);
criterion_main!(benches);
