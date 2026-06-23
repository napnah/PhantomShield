//! 融合批处理三方模拟驱动（PyO3 高速路径）。
//!
//! 不走 Comm 通道：在切片上直接调用各协议「纯数学核心」，逐元素独立，
//! 用 rayon 并行。每个元素的掩码由两套 PRG（共享 `prg0` / HP `asprg`）按
//! 「计数器 = 元素下标 × 步长」无状态生成（`raw64_at`），既确定可复现又可并行。
//!
//! 返回两方输出份额（Python 端 `s0 + s1` 重构并验证）。

use rayon::prelude::*;

use crate::prg::{real_from, unit_from, PrgSync};
use crate::protocols::exp::exp_core;
use crate::protocols::gelu::gelu_core;
use crate::protocols::multiply::multiply_core;
use crate::protocols::sigmoid::sigmoid_core;
use crate::protocols::softmax::softmax_core;
use crate::protocols::MOD;

/// 并行生成 `count` 个 PRG 原始 64 位字（对齐 Python `next()` 序列）。
pub fn prg_next_batch(seed: &[u8; 16], count: usize) -> Vec<u64> {
    let prg = PrgSync::new(seed);
    (0..count as u64)
        .into_par_iter()
        .map(|c| prg.raw64_at(c))
        .collect()
}

/// 批量整数环乘法。返回 (s0, s1)，s0[i]+s1[i] = x[i]*y[i] (mod 2^64)。
pub fn multiply_batch(
    x0: &[u64],
    x1: &[u64],
    y0: &[u64],
    y1: &[u64],
    seed_shared: &[u8; 16],
    seed_hp: &[u8; 16],
) -> (Vec<u64>, Vec<u64>) {
    let n = x0.len();
    let prg0 = PrgSync::new(seed_shared);
    let asprg = PrgSync::new(seed_hp);
    (0..n)
        .into_par_iter()
        .map(|i| {
            let r_x = prg0.raw64_at(2 * i as u64);
            let r_y = prg0.raw64_at(2 * i as u64 + 1);
            let s0 = asprg.raw64_at(i as u64);
            multiply_core(x0[i], x1[i], y0[i], y1[i], r_x, r_y, s0)
        })
        .unzip()
}

/// 批量安全指数。返回 (e0, e1)，e0[i]+e1[i] ≈ exp(x[i])。
pub fn exp_batch(
    x0: &[f64],
    x1: &[f64],
    seed_shared: &[u8; 16],
    seed_hp: &[u8; 16],
) -> (Vec<f64>, Vec<f64>) {
    let n = x0.len();
    let prg0 = PrgSync::new(seed_shared);
    let asprg = PrgSync::new(seed_hp);
    (0..n)
        .into_par_iter()
        .map(|i| {
            let x = x0[i] + x1[i];
            let r = real_from(prg0.raw64_at(i as u64), MOD);
            let u = unit_from(asprg.raw64_at(i as u64));
            exp_core(x, r, u)
        })
        .unzip()
}

/// 批量安全 Sigmoid。返回 (s0, s1)，s0[i]+s1[i] ≈ sigmoid(z[i])。
pub fn sigmoid_batch(
    z0: &[f64],
    z1: &[f64],
    seed_shared: &[u8; 16],
    seed_hp: &[u8; 16],
) -> (Vec<f64>, Vec<f64>) {
    let n = z0.len();
    let prg0 = PrgSync::new(seed_shared);
    let asprg = PrgSync::new(seed_hp);
    (0..n)
        .into_par_iter()
        .map(|i| {
            let z = z0[i] + z1[i];
            let r = real_from(prg0.raw64_at(2 * i as u64), MOD);
            let t = real_from(prg0.raw64_at(2 * i as u64 + 1), MOD);
            let u = unit_from(asprg.raw64_at(i as u64));
            sigmoid_core(z, r, u, t)
        })
        .unzip()
}

/// 批量安全 GeLU。返回 (g0, g1)，g0[i]+g1[i] ≈ gelu(x[i])。
pub fn gelu_batch(
    x0: &[f64],
    x1: &[f64],
    seed_shared: &[u8; 16],
    seed_hp: &[u8; 16],
) -> (Vec<f64>, Vec<f64>) {
    let n = x0.len();
    let prg0 = PrgSync::new(seed_shared);
    let asprg = PrgSync::new(seed_hp);
    (0..n)
        .into_par_iter()
        .map(|i| {
            let x = x0[i] + x1[i];
            let r = real_from(prg0.raw64_at(2 * i as u64), MOD);
            let t = real_from(prg0.raw64_at(2 * i as u64 + 1), MOD);
            let u = unit_from(asprg.raw64_at(2 * i as u64));
            let u_mul = unit_from(asprg.raw64_at(2 * i as u64 + 1));
            gelu_core(x, r, u, t, u_mul)
        })
        .unzip()
}

/// 批量安全 Softmax（按行）。输入展平的 `n×k` 行主序份额，返回展平的两方份额。
pub fn softmax_batch(
    x0: &[f64],
    x1: &[f64],
    n: usize,
    k: usize,
    seed_shared: &[u8; 16],
    seed_hp: &[u8; 16],
) -> (Vec<f64>, Vec<f64>) {
    let prg0 = PrgSync::new(seed_shared);
    let asprg = PrgSync::new(seed_hp);

    let mut out0 = vec![0.0f64; n * k];
    let mut out1 = vec![0.0f64; n * k];

    // 按行并行：每行独立读写自己的 k 长切片
    out0.par_chunks_mut(k)
        .zip(out1.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, (o0, o1))| {
            let base = i * k;
            let xs: Vec<f64> = (0..k).map(|j| x0[base + j] + x1[base + j]).collect();
            let stride_p = (k + 1) as u64;
            let rs: Vec<f64> = (0..k)
                .map(|j| real_from(prg0.raw64_at(i as u64 * stride_p + j as u64), MOD))
                .collect();
            let us: Vec<f64> = (0..k)
                .map(|j| unit_from(asprg.raw64_at(i as u64 * k as u64 + j as u64)))
                .collect();
            let t = real_from(prg0.raw64_at(i as u64 * stride_p + k as u64), MOD);
            softmax_core(&xs, &rs, &us, t, o0, o1);
        });

    (out0, out1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s0() -> [u8; 16] {
        let mut s = [0u8; 16];
        for (i, b) in s.iter_mut().enumerate() {
            *b = i as u8;
        }
        s
    }
    fn s1() -> [u8; 16] {
        let mut s = [0u8; 16];
        for (i, b) in s.iter_mut().enumerate() {
            *b = (i + 16) as u8;
        }
        s
    }

    #[test]
    fn exp_batch_ok() {
        let x0 = vec![1.0, -3.0, 5.0];
        let x1 = vec![0.5, 2.0, -1.0];
        let (e0, e1) = exp_batch(&x0, &x1, &s0(), &s1());
        for i in 0..3 {
            let want = (x0[i] + x1[i]).exp();
            assert!((e0[i] + e1[i] - want).abs() / want < 1e-9);
        }
    }

    #[test]
    fn softmax_batch_rows_sum_to_one() {
        let n = 2;
        let k = 4;
        let x0 = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
        let x1 = vec![0.0; n * k];
        let (o0, o1) = softmax_batch(&x0, &x1, n, k, &s0(), &s1());
        for i in 0..n {
            let sum: f64 = (0..k).map(|j| o0[i * k + j] + o1[i * k + j]).sum();
            assert!((sum - 1.0).abs() < 1e-9, "行 {i} 概率和 {sum}");
        }
    }
}
