//! Π_gelu：安全 GeLU（对齐 Python `protocols/gelu.py` 中 GeLU 部分）。
//!
//! GeLU(x) = x · sigmoid(1.702·x)。先本地缩放，复用 Π_sigmoid，再做一次实数域
//! 安全乘法。融合模拟中重构值等于 x·sigmoid(1.702x)，按 HP 随机 u_mul 拆成两份额。

use super::sigmoid::sigmoid_value;
use super::GELU_COEF;

/// 纯数学核心：输入真实 `x`、sigmoid 内部掩码 `r`/`u`/`t`、实数乘法分发随机 `u_mul`，
/// 返回 (g0, g1)，满足 g0 + g1 ≈ GeLU(x)。
#[inline]
pub fn gelu_core(x: f64, r: f64, u: f64, t: f64, u_mul: f64) -> (f64, f64) {
    let sig = sigmoid_value(GELU_COEF * x, r, u, t);
    let result = x * sig;
    let g0 = u_mul * result;
    let g1 = result - g0;
    (g0, g1)
}

/// 便捷：返回重构后的 GeLU(x)。
#[inline]
pub fn gelu_value(x: f64, r: f64, u: f64, t: f64, u_mul: f64) -> f64 {
    let (g0, g1) = gelu_core(x, r, u, t, u_mul);
    g0 + g1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(x: f64) -> f64 {
        x * (1.0 / (1.0 + (-(GELU_COEF * x)).exp()))
    }

    #[test]
    fn gelu_matches_plain() {
        for &x in &[-5.0, -1.0, 0.0, 0.7, 3.0, 5.0] {
            let got = gelu_value(x, 60.0, 0.22, 9.0, 0.5);
            assert!((got - plain(x)).abs() < 1e-9, "x={x}");
        }
    }
}
