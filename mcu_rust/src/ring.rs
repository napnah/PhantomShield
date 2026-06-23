//! Z_{2^64} 环运算。
//!
//! Python 端用大整数后 `% (2**64)`；Rust 的 `u64` 回绕（wrapping）加/乘
//! 恰好等价于 mod 2^64，且 mod 2^64 是环同态，因此各修正项可逐项 wrapping
//! 计算，结果与 Python 完全一致。

/// 环模数 L = 2^64（仅作文档标注；u64 回绕即 mod L）。
pub const RING_BITS: u32 = 64;

#[inline]
pub fn add(a: u64, b: u64) -> u64 {
    a.wrapping_add(b)
}

#[inline]
pub fn sub(a: u64, b: u64) -> u64 {
    a.wrapping_sub(b)
}

#[inline]
pub fn mul(a: u64, b: u64) -> u64 {
    a.wrapping_mul(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping_matches_mod_2_64() {
        // (2^64 - 1) * 2 mod 2^64 = 2^64 - 2
        assert_eq!(mul(u64::MAX, 2), u64::MAX - 1);
        assert_eq!(add(u64::MAX, 3), 2);
        assert_eq!(sub(1, 3), u64::MAX - 1);
    }
}
