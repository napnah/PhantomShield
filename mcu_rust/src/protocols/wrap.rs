//! Wrap 检测（对齐 Python `protocols/wrap_detect.py`）。
//!
//! 目标：w = floor((x + r) / M)。在 |x| < M 假设下 x+r ∈ (-M, 2M)，故 w ∈ {-1,0,1}，
//! 由两次符号比较得到：w = [x+r >= M] - [x+r < 0]。
//!
//! Python 通过「乘性正掩码 + HP 求和判号」实现两次比较；由于乘性正掩码不改变
//! 符号、且融合模拟中可直接得到 x+r，故纯核心直接对真实和判号（结果与协议一致）。

/// 纯数学核心：给定真实 `x`、公开掩码 `r`、模 `m`，返回 wrap 值 ∈ {-1,0,1}。
#[inline]
pub fn wrap_core(x: f64, r: f64, m: f64) -> i64 {
    let s = x + r;
    let ge_hi = (s >= m) as i64; // [x+r >= M]
    let lt_lo = (s < 0.0) as i64; // [x+r < 0]
    ge_hi - lt_lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_three_cases() {
        let m = 256.0;
        assert_eq!(wrap_core(10.0, 5.0, m), 0); // 15 ∈ [0,256)
        assert_eq!(wrap_core(200.0, 100.0, m), 1); // 300 ∈ [256,512)
        assert_eq!(wrap_core(-100.0, 5.0, m), -1); // -95 < 0
    }
}
