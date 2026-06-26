use crate::channel::{HpComm, Msg, PartyComm};
use crate::prg::PrgSync;
use crate::protocols::{GELU_COEF, MOD};
use crate::ring::{add, mul, sub};

const SIGN_SCALE: f64 = (1u64 << 20) as f64;
const WRAP_FIXED_SCALE_BITS: u32 = 24;
const WRAP_FIXED_SCALE: f64 = (1u64 << WRAP_FIXED_SCALE_BITS) as f64;
const WRAP_FIXED_LX: u32 = 48;

/// Engineering Sign protocol for real-valued shares.
///
/// This is a true P0/P1/HP communication protocol and the interface mirrors the
/// role of Protocol 3 in the paper: it returns sign(z) as a public bit. The
/// internal comparison is the simplified positive multiplicative masking used
/// by the Python reference implementation, not the full Bicoptor probabilistic
/// truncation construction cited by the paper.
pub fn party_sign_ge_zero<C: PartyComm>(
    share_z: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> u8 {
    let alpha = 1.0 + prg0.next_real(SIGN_SCALE);
    comm.send_to_hp(Msg::Real(alpha * share_z));
    comm.recv_from_hp().as_bit()
}

pub fn hp_sign_ge_zero<C: HpComm>(comm: &C) {
    let z0 = comm.recv_from_p0().as_real();
    let z1 = comm.recv_from_p1().as_real();
    let bit = if z0 + z1 >= 0.0 { 1 } else { 0 };
    comm.send_to_p0(Msg::Bit(bit));
    comm.send_to_p1(Msg::Bit(bit));
}

/// Bicoptor-style integer DReLU/sign protocol from Algorithm 2.
///
/// Input shares are in Z_{2^64}. The represented value must be in
/// [0, 2^lx) for non-negative values or (2^64 - 2^lx, 2^64) for negative
/// values. The public output bit is 1 for non-negative values, 0 otherwise.
pub fn party_sign_bicoptor<C: PartyComm>(
    id: u8,
    share_x: u64,
    lx: u32,
    prg0: &mut PrgSync,
    comm: &C,
) -> u8 {
    assert!(lx > 0 && lx < 63, "lx must be in 1..63");
    let input_mask = prg0.next();
    let share_x = if id == 0 {
        add(share_x, input_mask)
    } else {
        sub(share_x, input_mask)
    };
    let t = (prg0.next() & 1) as u8;
    let x_share = if t == 0 { share_x } else { share_x.wrapping_neg() };

    let mut masks = Vec::with_capacity((lx + 2) as usize);
    for _ in 0..(lx + 2) {
        masks.push(prg0.next() | 1); // odd masks are units in Z_2^64
    }

    let u_star_const = if t == 0 { 1u64 } else { 0u64.wrapping_sub(1) };
    let u_star = if id == 0 { u_star_const } else { 0 };

    let mut u = Vec::with_capacity((lx + 1) as usize);
    u.push(x_share);
    for k in 1..=lx {
        u.push(trc_share(id, x_share, k));
    }

    let mut values = Vec::with_capacity((lx + 2) as usize);
    let public_one = if id == 0 { 1 } else { 0 };
    let v_star = sub(add(u_star, mul(3, u[0])), public_one);
    values.push(v_star);
    for i in 0..=lx as usize {
        let mut acc = 0u64;
        for val in u.iter().take(lx as usize + 1).skip(i) {
            acc = add(acc, *val);
        }
        values.push(sub(acc, public_one));
    }

    for (value, mask) in values.iter_mut().zip(masks.iter()) {
        *value = mul(*value, *mask);
    }
    shuffle_with_prg(&mut values, prg0);

    let mut reshared_values = Vec::with_capacity(values.len());
    for value in values {
        let reshare_mask = prg0.next();
        let reshared = if id == 0 {
            add(value, reshare_mask)
        } else {
            sub(value, reshare_mask)
        };
        reshared_values.push(reshared);
    }
    comm.send_to_hp(Msg::ShareVec(reshared_values));
    let dprime = comm.recv_from_hp().as_bit();
    if t == 0 { dprime } else { 1 - dprime }
}

pub fn hp_sign_bicoptor<C: HpComm>(lx: u32, comm: &C) {
    let count = (lx + 2) as usize;
    let w0_values = comm.recv_from_p0().into_share_vec();
    let w1_values = comm.recv_from_p1().into_share_vec();
    assert_eq!(w0_values.len(), count, "p0 sign-bicoptor vector length mismatch");
    assert_eq!(w1_values.len(), count, "p1 sign-bicoptor vector length mismatch");
    let mut has_zero = false;
    for (w0, w1) in w0_values.into_iter().zip(w1_values) {
        if add(w0, w1) == 0 {
            has_zero = true;
        }
    }
    let dprime = if has_zero { 1 } else { 0 };
    comm.send_to_p0(Msg::Bit(dprime));
    comm.send_to_p1(Msg::Bit(dprime));
}

fn trc_share(id: u8, share: u64, k: u32) -> u64 {
    if id == 0 {
        share >> k
    } else {
        (share.wrapping_neg() >> k).wrapping_neg()
    }
}

fn shuffle_with_prg(values: &mut [u64], prg0: &mut PrgSync) {
    for i in (1..values.len()).rev() {
        let j = (prg0.next() as usize) % (i + 1);
        values.swap(i, j);
    }
}

pub fn party_wrap<C: PartyComm>(
    id: u8,
    share_x: f64,
    r: f64,
    m: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> i64 {
    let ge_hi = party_sign_ge(id, share_x, m - r, prg0, comm);
    let ge_lo = party_sign_ge(id, share_x, -r, prg0, comm);
    ge_hi as i64 - (1 - ge_lo as i64)
}

pub fn hp_wrap<C: HpComm>(comm: &C) {
    hp_sign_ge(comm);
    hp_sign_ge(comm);
}

pub fn party_wrap_bicoptor<C: PartyComm>(
    id: u8,
    share_x: f64,
    r: f64,
    m: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> i64 {
    let ge_hi = party_sign_ge_fixed(id, share_x, m - r, prg0, comm);
    let ge_lo = party_sign_ge_fixed(id, share_x, -r, prg0, comm);
    ge_hi as i64 - (1 - ge_lo as i64)
}

pub fn hp_wrap_bicoptor<C: HpComm>(comm: &C) {
    hp_sign_bicoptor(WRAP_FIXED_LX, comm);
    hp_sign_bicoptor(WRAP_FIXED_LX, comm);
}

fn party_sign_ge_fixed<C: PartyComm>(
    id: u8,
    share_x: f64,
    threshold: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> u8 {
    let fixed_share = fixed_compare_share(id, share_x, threshold);
    party_sign_bicoptor(id, fixed_share, WRAP_FIXED_LX, prg0, comm)
}

fn fixed_compare_share(id: u8, share_x: f64, threshold: f64) -> u64 {
    let shifted = share_x - if id == 0 { threshold } else { 0.0 };
    let fixed = (shifted * WRAP_FIXED_SCALE).round() as i64;
    fixed as u64
}

fn party_sign_ge<C: PartyComm>(
    id: u8,
    share_x: f64,
    threshold: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> u8 {
    let z_i = share_x - if id == 0 { threshold } else { 0.0 };
    party_sign_ge_zero(z_i, prg0, comm)
}

fn hp_sign_ge<C: HpComm>(comm: &C) {
    hp_sign_ge_zero(comm)
}

fn party_exp_with_correction<C: PartyComm>(
    id: u8,
    share_x: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> (f64, f64) {
    let r = prg0.next_real(MOD);
    let masked = if id == 0 {
        (share_x + r).rem_euclid(MOD)
    } else {
        share_x.rem_euclid(MOD)
    };
    comm.send_to_hp(Msg::Real(masked));
    let s_i = comm.recv_from_hp().as_real();
    let w = party_wrap_bicoptor(id, share_x, r, MOD, prg0, comm);
    let correction = ((w as f64) * MOD - r).exp();
    (s_i * correction, correction)
}

pub fn party_exp<C: PartyComm>(id: u8, share_x: f64, prg0: &mut PrgSync, comm: &C) -> f64 {
    party_exp_with_correction(id, share_x, prg0, comm).0
}

fn hp_exp_masked<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) -> f64 {
    let m0 = comm.recv_from_p0().as_real();
    let m1 = comm.recv_from_p1().as_real();
    let r = (m0 + m1).rem_euclid(MOD);
    let e = r.exp();
    let u = asprg_p0.next_unit();
    let s0 = u * e;
    let s1 = e - s0;
    comm.send_to_p0(Msg::Real(s0));
    comm.send_to_p1(Msg::Real(s1));
    hp_wrap_bicoptor(comm);
    e
}

pub fn hp_exp<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) {
    hp_exp_masked(asprg_p0, comm);
}

pub fn party_sigmoid<C: PartyComm>(
    id: u8,
    share_z: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> f64 {
    let (e_i, correction) = party_exp_with_correction(id, share_z, prg0, comm);
    let d_i = e_i + if id == 0 { 1.0 } else { 0.0 };
    let t = prg0.next_real(MOD);
    let et = t.exp();
    comm.send_to_hp(Msg::Real(et * d_i));
    let fraction_share = comm.recv_from_hp().as_real();
    fraction_share * et * correction
}

pub fn hp_sigmoid<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) {
    let masked_numerator = hp_exp_masked(asprg_p0, comm);
    let u0 = comm.recv_from_p0().as_real();
    let u1 = comm.recv_from_p1().as_real();
    let denominator = u0 + u1;
    let fraction = masked_numerator / denominator;
    let u = asprg_p0.next_unit();
    let s0 = u * fraction;
    let s1 = fraction - s0;
    comm.send_to_p0(Msg::Real(s0));
    comm.send_to_p1(Msg::Real(s1));
}

pub fn party_softmax<C: PartyComm>(
    id: u8,
    shares: &[f64],
    target: usize,
    prg0: &mut PrgSync,
    comm: &C,
) -> f64 {
    party_softmax_all(id, shares, prg0, comm)[target]
}

pub fn party_softmax_all<C: PartyComm>(
    id: u8,
    shares: &[f64],
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<f64> {
    let mut exp_shares = Vec::with_capacity(shares.len());
    let mut corrections = Vec::with_capacity(shares.len());
    for &s in shares {
        let (e_i, correction) = party_exp_with_correction(id, s, prg0, comm);
        exp_shares.push(e_i);
        corrections.push(correction);
    }
    let t = prg0.next_real(MOD);
    let et = t.exp();
    let denom_share: f64 = exp_shares.iter().sum();
    comm.send_to_hp(Msg::Real(et * denom_share));
    corrections
        .into_iter()
        .map(|correction| comm.recv_from_hp().as_real() * et * correction)
        .collect()
}

pub fn hp_softmax<C: HpComm>(k: usize, asprg_p0: &mut PrgSync, comm: &C) {
    let mut masked_numerators = Vec::with_capacity(k);
    for _ in 0..k {
        masked_numerators.push(hp_exp_masked(asprg_p0, comm));
    }
    let u0 = comm.recv_from_p0().as_real();
    let u1 = comm.recv_from_p1().as_real();
    let denominator = u0 + u1;
    for masked_numerator in masked_numerators {
        let fraction = masked_numerator / denominator;
        let u = asprg_p0.next_unit();
        let s0 = u * fraction;
        let s1 = fraction - s0;
        comm.send_to_p0(Msg::Real(s0));
        comm.send_to_p1(Msg::Real(s1));
    }
}

pub fn party_gelu<C: PartyComm>(id: u8, share_x: f64, prg0: &mut PrgSync, comm: &C) -> f64 {
    let sig_i = party_sigmoid(id, GELU_COEF * share_x, prg0, comm);
    let r_a = prg0.next_real(MOD);
    let r_b = prg0.next_real(MOD);
    if id == 0 {
        comm.send_to_hp(Msg::RealPair(share_x + r_a, sig_i + r_b));
    } else {
        comm.send_to_hp(Msg::RealPair(share_x, sig_i));
    }
    let s_i = comm.recv_from_hp().as_real();
    let correction = if id == 0 {
        share_x * r_b + sig_i * r_a + r_a * r_b
    } else {
        share_x * r_b + sig_i * r_a
    };
    s_i - correction
}

pub fn hp_gelu<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) {
    hp_sigmoid(asprg_p0, comm);
    let (a0, b0) = comm.recv_from_p0().as_real_pair();
    let (a1, b1) = comm.recv_from_p1().as_real_pair();
    let product = (a0 + a1) * (b0 + b1);
    let u = asprg_p0.next_unit();
    let s0 = u * product;
    let s1 = product - s0;
    comm.send_to_p0(Msg::Real(s0));
    comm.send_to_p1(Msg::Real(s1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::make_mock;
    use std::thread;

    const S0: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    const S1: [u8; 16] = [
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];

    fn run2<F0, F1, FH>(f0: F0, f1: F1, fh: FH) -> (f64, f64)
    where
        F0: FnOnce(crate::channel::PartyEndpoint) -> f64 + Send + 'static,
        F1: FnOnce(crate::channel::PartyEndpoint) -> f64 + Send + 'static,
        FH: FnOnce(crate::channel::HpEndpoint) + Send + 'static,
    {
        let (p0, p1, hp) = make_mock();
        let t0 = thread::spawn(move || f0(p0));
        let t1 = thread::spawn(move || f1(p1));
        let th = thread::spawn(move || fh(hp));
        let o0 = t0.join().unwrap();
        let o1 = t1.join().unwrap();
        th.join().unwrap();
        (o0, o1)
    }

    #[test]
    fn real_exp_comm_matches_plain() {
        let x = 1.25;
        let x0 = 7.0;
        let x1 = x - x0;
        let (o0, o1) = run2(
            move |comm| party_exp(0, x0, &mut PrgSync::new(&S0), &comm),
            move |comm| party_exp(1, x1, &mut PrgSync::new(&S0), &comm),
            move |comm| hp_exp(&mut PrgSync::new(&S1), &comm),
        );
        assert!((o0 + o1 - x.exp()).abs() < 1e-10);
    }

    #[test]
    fn real_sign_comm_matches_plain() {
        let x: f64 = -3.5;
        let x0 = 10.0;
        let x1 = x - x0;
        let (o0, o1) = run2(
            move |comm| party_sign_ge_zero(x0, &mut PrgSync::new(&S0), &comm) as f64,
            move |comm| party_sign_ge_zero(x1, &mut PrgSync::new(&S0), &comm) as f64,
            move |comm| hp_sign_ge_zero(&comm),
        );
        assert_eq!(o0 as u8, 0);
        assert_eq!(o1 as u8, 0);
    }

    #[test]
    fn real_bicoptor_sign_comm_matches_plain() {
        let lx = 16;
        for &x in &[1234i64, -5678i64, 1i64, -1i64] {
            let x0 = 0x1234_5678_ABCD_EF01u64;
            let x1 = (x as u64).wrapping_sub(x0);
            let (o0, o1) = run2(
                move |comm| party_sign_bicoptor(0, x0, lx, &mut PrgSync::new(&S0), &comm) as f64,
                move |comm| party_sign_bicoptor(1, x1, lx, &mut PrgSync::new(&S0), &comm) as f64,
                move |comm| hp_sign_bicoptor(lx, &comm),
            );
            let want = if x >= 0 { 1 } else { 0 };
            assert_eq!(o0 as u8, want, "x={x}");
            assert_eq!(o1 as u8, want, "x={x}");
        }
    }

    #[test]
    fn real_bicoptor_sign_comm_matches_fixed_wrap_input() {
        let lx = WRAP_FIXED_LX;
        let x = (44.0 * WRAP_FIXED_SCALE).round() as i64;
        let x0 = (-175.0 * WRAP_FIXED_SCALE).round() as i64 as u64;
        let x1 = (x as u64).wrapping_sub(x0);
        let (o0, o1) = run2(
            move |comm| party_sign_bicoptor(0, x0, lx, &mut PrgSync::new(&S0), &comm) as f64,
            move |comm| party_sign_bicoptor(1, x1, lx, &mut PrgSync::new(&S0), &comm) as f64,
            move |comm| hp_sign_bicoptor(lx, &comm),
        );
        assert_eq!(o0 as u8, 1);
        assert_eq!(o1 as u8, 1);
    }

    #[test]
    fn real_bicoptor_sign_comm_matches_two_wrap_inputs_in_sequence() {
        let lx = WRAP_FIXED_LX;
        let x0_hi = (-175.0 * WRAP_FIXED_SCALE).round() as i64 as u64;
        let x1_hi = ((44.0 * WRAP_FIXED_SCALE).round() as i64 as u64).wrapping_sub(x0_hi);
        let x0_lo = (81.0 * WRAP_FIXED_SCALE).round() as i64 as u64;
        let x1_lo = ((300.0 * WRAP_FIXED_SCALE).round() as i64 as u64).wrapping_sub(x0_lo);
        let (p0, p1, hp) = make_mock();
        let t0 = thread::spawn(move || {
            let mut prg = PrgSync::new(&S0);
            (
                party_sign_bicoptor(0, x0_hi, lx, &mut prg, &p0),
                party_sign_bicoptor(0, x0_lo, lx, &mut prg, &p0),
            )
        });
        let t1 = thread::spawn(move || {
            let mut prg = PrgSync::new(&S0);
            (
                party_sign_bicoptor(1, x1_hi, lx, &mut prg, &p1),
                party_sign_bicoptor(1, x1_lo, lx, &mut prg, &p1),
            )
        });
        let th = thread::spawn(move || {
            hp_sign_bicoptor(lx, &hp);
            hp_sign_bicoptor(lx, &hp);
        });
        let o0 = t0.join().unwrap();
        let o1 = t1.join().unwrap();
        th.join().unwrap();
        assert_eq!(o0, (1, 1));
        assert_eq!(o1, (1, 1));
    }

    #[test]
    fn real_wrap_comm_matches_plain() {
        let x: f64 = 200.0;
        let r = 100.0;
        let x0 = -19.0;
        let x1 = x - x0;
        let (o0, o1) = run2(
            move |comm| party_wrap(0, x0, r, MOD, &mut PrgSync::new(&S0), &comm) as f64,
            move |comm| party_wrap(1, x1, r, MOD, &mut PrgSync::new(&S0), &comm) as f64,
            move |comm| hp_wrap(&comm),
        );
        assert_eq!(o0 as i64, 1);
        assert_eq!(o1 as i64, 1);
    }

    #[test]
    fn real_wrap_bicoptor_comm_matches_plain() {
        for &(x, r, want) in &[(10.0, 5.0, 0), (200.0, 100.0, 1), (-100.0, 5.0, -1)] {
            let x0 = -19.0;
            let x1 = x - x0;
            let (o0, o1) = run2(
                move |comm| {
                    party_wrap_bicoptor(0, x0, r, MOD, &mut PrgSync::new(&S0), &comm) as f64
                },
                move |comm| {
                    party_wrap_bicoptor(1, x1, r, MOD, &mut PrgSync::new(&S0), &comm) as f64
                },
                move |comm| hp_wrap_bicoptor(&comm),
            );
            assert_eq!(o0 as i64, want, "x={x}, r={r}");
            assert_eq!(o1 as i64, want, "x={x}, r={r}");
        }
    }

    #[test]
    fn real_sigmoid_comm_matches_plain() {
        let x: f64 = -2.0;
        let x0 = 3.0;
        let x1 = x - x0;
        let (o0, o1) = run2(
            move |comm| party_sigmoid(0, x0, &mut PrgSync::new(&S0), &comm),
            move |comm| party_sigmoid(1, x1, &mut PrgSync::new(&S0), &comm),
            move |comm| hp_sigmoid(&mut PrgSync::new(&S1), &comm),
        );
        let want = 1.0 / (1.0 + (-x).exp());
        assert!((o0 + o1 - want).abs() < 1e-10);
    }

    #[test]
    fn real_gelu_comm_matches_plain() {
        let x: f64 = 0.75;
        let x0 = -4.0;
        let x1 = x - x0;
        let (o0, o1) = run2(
            move |comm| party_gelu(0, x0, &mut PrgSync::new(&S0), &comm),
            move |comm| party_gelu(1, x1, &mut PrgSync::new(&S0), &comm),
            move |comm| hp_gelu(&mut PrgSync::new(&S1), &comm),
        );
        let want = x * (1.0 / (1.0 + (-(GELU_COEF * x)).exp()));
        assert!((o0 + o1 - want).abs() < 1e-10);
    }

    #[test]
    fn real_softmax_comm_matches_plain() {
        let xs = [1.0, -2.0, 3.0, 0.5];
        let xs0 = [7.0, -4.0, 0.25, 2.0];
        let xs1: Vec<f64> = xs.iter().zip(xs0.iter()).map(|(x, x0)| x - x0).collect();
        let (p0, p1, hp) = make_mock();
        let t0 = thread::spawn(move || party_softmax_all(0, &xs0, &mut PrgSync::new(&S0), &p0));
        let t1 = thread::spawn(move || party_softmax_all(1, &xs1, &mut PrgSync::new(&S0), &p1));
        let th = thread::spawn(move || hp_softmax(xs.len(), &mut PrgSync::new(&S1), &hp));
        let o0 = t0.join().unwrap();
        let o1 = t1.join().unwrap();
        th.join().unwrap();
        let denom: f64 = xs.iter().map(|x| f64::exp(*x)).sum();
        for i in 0..xs.len() {
            let want = xs[i].exp() / denom;
            assert!((o0[i] + o1[i] - want).abs() < 1e-10);
        }
    }
}
