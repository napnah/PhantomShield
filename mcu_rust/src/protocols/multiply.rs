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

pub fn party_multiply_batch<C: PartyComm>(
    id: u8,
    share_x: &[u64],
    share_y: &[u64],
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<u64> {
    assert_eq!(share_x.len(), share_y.len(), "multiply batch length mismatch");
    let mut masked_x = Vec::with_capacity(share_x.len());
    let mut masked_y = Vec::with_capacity(share_y.len());
    let mut corrections = Vec::with_capacity(share_x.len());

    for (&x, &y) in share_x.iter().zip(share_y.iter()) {
        let r_x = prg0.next();
        let r_y = prg0.next();
        if id == 0 {
            masked_x.push(add(x, r_x));
            masked_y.push(add(y, r_y));
            corrections.push(add(add(mul(x, r_y), mul(y, r_x)), mul(r_x, r_y)));
        } else {
            masked_x.push(x);
            masked_y.push(y);
            corrections.push(add(mul(x, r_y), mul(y, r_x)));
        }
    }

    comm.send_to_hp(Msg::MulVecToHp {
        id,
        mx: masked_x,
        my: masked_y,
    });
    let hp_shares = comm.recv_from_hp().into_share_vec();
    assert_eq!(
        hp_shares.len(),
        corrections.len(),
        "HP multiply batch share length mismatch"
    );
    corrections
        .into_iter()
        .zip(hp_shares)
        .map(|(correction, share)| sub(share, correction))
        .collect()
}

/// HP 通过 Comm 处理一次乘法。
pub fn hp_multiply<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) {
    let (msg0, msg1) = comm.recv_from_parties();
    let (_, mx0, my0) = msg0.as_mul();
    let (_, mx1, my1) = msg1.as_mul();
    let mx = add(mx0, mx1);
    let my = add(my0, my1);
    let product = mul(mx, my);
    let s0 = asprg_p0.next();
    let s1 = sub(product, s0);
    comm.send_to_parties(Msg::Share(s0), Msg::Share(s1));
}

pub fn hp_multiply_batch<C: HpComm>(n: usize, asprg_p0: &mut PrgSync, comm: &C) {
    let (msg0, msg1) = comm.recv_from_parties();
    let (_, mx0, my0) = msg0.as_mul_vec();
    let (_, mx1, my1) = msg1.as_mul_vec();
    assert_eq!(mx0.len(), n, "p0 multiply batch x length mismatch");
    assert_eq!(mx1.len(), n, "p1 multiply batch x length mismatch");
    assert_eq!(my0.len(), n, "p0 multiply batch y length mismatch");
    assert_eq!(my1.len(), n, "p1 multiply batch y length mismatch");
    let mut out0 = Vec::with_capacity(n);
    let mut out1 = Vec::with_capacity(n);
    for i in 0..n {
        let mx = add(mx0[i], mx1[i]);
        let my = add(my0[i], my1[i]);
        let product = mul(mx, my);
        let s0 = asprg_p0.next();
        let s1 = sub(product, s0);
        out0.push(s0);
        out1.push(s1);
    }
    comm.send_to_parties(Msg::ShareVec(out0), Msg::ShareVec(out1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::make_mock;
    use std::thread;

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

    #[test]
    fn batch_comm_matches_plaintext() {
        const SEED_SHARED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        const SEED_HP: [u8; 16] = [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let x = [3u64, 7, 11, 19, u64::MAX - 4];
        let y = [5u64, 13, 17, 23, 9];
        let x0 = [100u64, 101, 102, 103, 104];
        let y0 = [200u64, 201, 202, 203, 204];
        let x1: Vec<u64> = x.iter().zip(x0.iter()).map(|(&v, &s)| sub(v, s)).collect();
        let y1: Vec<u64> = y.iter().zip(y0.iter()).map(|(&v, &s)| sub(v, s)).collect();
        let (p0, p1, hp) = make_mock();
        let t0 = thread::spawn(move || {
            party_multiply_batch(0, &x0, &y0, &mut PrgSync::new(&SEED_SHARED), &p0)
        });
        let t1 = thread::spawn(move || {
            party_multiply_batch(1, &x1, &y1, &mut PrgSync::new(&SEED_SHARED), &p1)
        });
        let th =
            thread::spawn(move || hp_multiply_batch(x.len(), &mut PrgSync::new(&SEED_HP), &hp));

        let o0 = t0.join().unwrap();
        let o1 = t1.join().unwrap();
        th.join().unwrap();
        for i in 0..x.len() {
            assert_eq!(add(o0[i], o1[i]), mul(x[i], y[i]), "i={i}");
        }
    }
}
