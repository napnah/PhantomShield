//! Π_sigmoid：安全 Sigmoid（对齐 Python `protocols/gelu.py` 中 Sigmoid 部分）。
//!
//! sigmoid(z) = e^z / (1 + e^z)。复用 Π_exp 的数值路径得 e^z（拆为两份额），
//! 分母 (1+e^z) 用 e^t 掩码后公开：D = e^t·(1+e^z)，份额本地相除。

use super::exp::exp_core;

/// 纯数学核心：输入真实 `z`、exp 掩码 `r`、HP 随机 `u`、分母掩码指数 `t`，
/// 返回 (sig0, sig1)，满足 sig0 + sig1 ≈ sigmoid(z)。
#[inline]
pub fn sigmoid_core(z: f64, r: f64, u: f64, t: f64) -> (f64, f64) {
    let (e0, e1) = exp_core(z, r, u); // e0+e1 = e^z
    let et = t.exp();
    // D_pub = e^t·(1+e^z) = e^t·((e0 + 1) + e1)
    let d_pub = et * ((e0 + 1.0) + e1);
    (e0 * et / d_pub, e1 * et / d_pub)
}

/// 便捷：返回重构后的 sigmoid(z)。
#[inline]
pub fn sigmoid_value(z: f64, r: f64, u: f64, t: f64) -> f64 {
    let (s0, s1) = sigmoid_core(z, r, u, t);
    s0 + s1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    #[test]
    fn sigmoid_matches_plain() {
        for &z in &[-8.0, -2.0, 0.0, 1.5, 8.0] {
            let got = sigmoid_value(z, 77.0, 0.4, 30.0);
            assert!((got - plain(z)).abs() < 1e-10, "z={z}");
        }
    }
}
