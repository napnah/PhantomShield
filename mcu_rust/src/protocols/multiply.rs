//! Π_mul：整数环安全乘法（对齐 Python `protocols/multiply.py`）。
//!
//! 容斥恒等式：
//! ```text
//! xy = (x+r_x)(y+r_y) - x*r_y - y*r_x - r_x*r_y
//! P0 修正三项：x0*r_y + y0*r_x + r_x*r_y
//! P1 修正两项：x1*r_y + y1*r_x
//! ```
//! 全部在 Z_{2^64}（u64 回绕）上运算，与 Python `% (2**64)` 逐位一致。

use crate::channel::{HpComm, Msg, PartyComm};
use crate::prg::PrgSync;
use crate::ring::{add, mul, sub};

/// 纯数学核心：给定双方份额与公开掩码 + HP 随机份额 s0，返回两方输出份额。
///
/// `r_x, r_y` 为三方同步公开掩码；`s0` 为 HP 用 ASPRG 生成的分发随机数。
#[inline]
pub fn multiply_core(
    x0: u64,
    x1: u64,
    y0: u64,
    y1: u64,
    r_x: u64,
    r_y: u64,
    s0: u64,
) -> (u64, u64) {
    // HP 视角：重建掩码值并求积
    let mx = add(add(x0, r_x), x1); // x + r_x
    let my = add(add(y0, r_y), y1); // y + r_y
    let product = mul(mx, my);
    let s1 = sub(product, s0);

    // 各方去掩码修正
    let c0 = add(add(mul(x0, r_y), mul(y0, r_x)), mul(r_x, r_y));
    let c1 = add(mul(x1, r_y), mul(y1, r_x));

    let out0 = sub(s0, c0);
    let out1 = sub(s1, c1);
    (out0, out1)
}

// --------------------------------------------------------------------------- //
// Comm 抽象路径（忠实复现 2 轮通信，便于未来接 socket）
// --------------------------------------------------------------------------- //

/// P_i 通过 Comm 执行乘法，返回本方输出份额。
pub fn party_multiply<C: PartyComm>(
    id: u8,
    share_x: u64,
    share_y: u64,
    prg0: &mut PrgSync,
    comm: &C,
) -> u64 {
    let r_x = prg0.next();
    let r_y = prg0.next();

    if id == 0 {
        comm.send_to_hp(Msg::MulToHp {
            id: 0,
            mx: add(share_x, r_x),
            my: add(share_y, r_y),
        });
    } else {
        comm.send_to_hp(Msg::MulToHp {
            id: 1,
            mx: share_x,
            my: share_y,
        });
    }

    let s_i = comm.recv_from_hp().as_share();

    let correction = if id == 0 {
        add(add(mul(share_x, r_y), mul(share_y, r_x)), mul(r_x, r_y))
    } else {
        add(mul(share_x, r_y), mul(share_y, r_x))
    };
    sub(s_i, correction)
}

/// HP 通过 Comm 处理一次乘法。
pub fn hp_multiply<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) {
    let (_, mx0, my0) = comm.recv_from_p0().as_mul();
    let (_, mx1, my1) = comm.recv_from_p1().as_mul();
    let mx = add(mx0, mx1);
    let my = add(my0, my1);
    let product = mul(mx, my);
    let s0 = asprg_p0.next();
    let s1 = sub(product, s0);
    comm.send_to_p0(Msg::Share(s0));
    comm.send_to_p1(Msg::Share(s1));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_matches_plaintext() {
        let x: u64 = 12345;
        let y: u64 = 67890;
        let x0: u64 = 999999;
        let x1 = x.wrapping_sub(x0);
        let y0: u64 = 888888;
        let y1 = y.wrapping_sub(y0);
        let (o0, o1) = multiply_core(x0, x1, y0, y1, 0xDEAD, 0xBEEF, 0x1234);
        assert_eq!(add(o0, o1), x.wrapping_mul(y));
    }
}
