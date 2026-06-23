//! Π_softmax：安全 Softmax（对齐 Python `protocols/softmax.py`）。
//!
//! softmax(x_m) = e^{x_m} / Σ_j e^{x_j}。对每个分量复用 Π_exp 数值路径，
//! 分母用 e^t 掩码后公开：D = e^t·Σ_j e^{x_j}，各份额本地相除。

use super::exp::exp_core;

/// 纯数学核心：输入真实向量 `xs`（长度 k）、每分量 exp 掩码 `rs`/HP 随机 `us`、
/// 分母掩码指数 `t`，写出两方份额向量 `out0`/`out1`（长度 k）。
///
/// 满足 out0[m] + out1[m] ≈ softmax(xs)[m]。
#[inline]
pub fn softmax_core(xs: &[f64], rs: &[f64], us: &[f64], t: f64, out0: &mut [f64], out1: &mut [f64]) {
    let k = xs.len();
    debug_assert_eq!(rs.len(), k);
    debug_assert_eq!(us.len(), k);
    debug_assert_eq!(out0.len(), k);
    debug_assert_eq!(out1.len(), k);

    let et = t.exp();
    let mut sum_e = 0.0f64;
    // 先算各分量 e^{x_j} 的两份额，暂存到 out0/out1
    for j in 0..k {
        let (e0, e1) = exp_core(xs[j], rs[j], us[j]);
        out0[j] = e0;
        out1[j] = e1;
        sum_e += e0 + e1;
    }
    let d_pub = et * sum_e; // = e^t · Σ_j e^{x_j}
    for j in 0..k {
        out0[j] = out0[j] * et / d_pub;
        out1[j] = out1[j] * et / d_pub;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(xs: &[f64]) -> Vec<f64> {
        let mx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = xs.iter().map(|v| (v - mx).exp()).collect();
        let s: f64 = exps.iter().sum();
        exps.iter().map(|e| e / s).collect()
    }

    #[test]
    fn softmax_matches_plain() {
        let xs = [1.0, -2.0, 3.0, 0.5, -1.5, 2.2, 4.0, -3.0];
        let k = xs.len();
        let rs = vec![50.0; k];
        let us = vec![0.31; k];
        let mut o0 = vec![0.0; k];
        let mut o1 = vec![0.0; k];
        softmax_core(&xs, &rs, &us, 12.0, &mut o0, &mut o1);
        let want = plain(&xs);
        for m in 0..k {
            let got = o0[m] + o1[m];
            assert!((got - want[m]).abs() < 1e-10, "m={m}: {got} vs {}", want[m]);
        }
    }
}
