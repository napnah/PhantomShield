//! Π_exp：安全指数（对齐 Python `protocols/exponential.py`）。
//!
//! 恒等式：e^x = e^R · e^(w·M - r)，R = (x+r) mod M，w = floor((x+r)/M)。
//! 数值路径逐步复刻 Python，以保证两端舍入误差特性一致：
//!   R 用 rem_euclid（与 Python 浮点 `%` 同为非负余数）；
//!   E = exp(R) 拆成两个正份额 s0=u*E, s1=E-s0；
//!   修正因子 corr = exp(w*M - r) 同乘到两份额。

use super::wrap::wrap_core;
use super::MOD;

/// 纯数学核心：输入真实 `x`、公开掩码 `r`、HP 随机 `u∈[0,1)`，返回 (e0, e1)，
/// 满足 e0 + e1 ≈ e^x。
#[inline]
pub fn exp_core(x: f64, r: f64, u: f64) -> (f64, f64) {
    let big_r = (x + r).rem_euclid(MOD); // R = (x+r) mod M ∈ [0, M)
    let e = big_r.exp(); // E = e^R
    let s0 = u * e;
    let s1 = e - s0;
    let w = wrap_core(x, r, MOD) as f64;
    let corr = (w * MOD - r).exp(); // e^(w·M - r)
    (s0 * corr, s1 * corr)
}

/// 便捷：返回重构后的 e^x（= e0 + e1）。供 sigmoid/softmax/gelu 复用同一数值路径。
#[inline]
pub fn exp_value(x: f64, r: f64, u: f64) -> f64 {
    let (e0, e1) = exp_core(x, r, u);
    e0 + e1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_libm() {
        for &x in &[-10.0, -3.5, 0.0, 1.0, 7.2, 10.0] {
            let got = exp_value(x, 123.4, 0.37);
            let want = x.exp();
            assert!((got - want).abs() / want.max(1e-12) < 1e-10, "x={x}: {got} vs {want}");
        }
    }
}
