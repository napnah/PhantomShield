use crate::channel::{HpComm, Msg, PartyComm};
use crate::prg::{unit_from, PrgSync};
use crate::protocols::{GELU_COEF, MOD};
use crate::ring::{add, mul, sub};
use rayon::prelude::*;

const SIGN_SCALE: f64 = (1u64 << 20) as f64;
const WRAP_FIXED_SCALE_BITS: u32 = 24;
const WRAP_FIXED_SCALE: f64 = (1u64 << WRAP_FIXED_SCALE_BITS) as f64;
const WRAP_FIXED_LX_DEFAULT: u32 = 33;
const EXP_PAR_MIN_DEFAULT: usize = 1_048_576;

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

pub fn party_sign_ge_zero_batch<C: PartyComm>(
    shares_z: &[f64],
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<u8> {
    let masked: Vec<f64> = shares_z
        .iter()
        .map(|&share_z| {
            let alpha = 1.0 + prg0.next_real(SIGN_SCALE);
            alpha * share_z
        })
        .collect();
    comm.send_to_hp(Msg::RealVec(masked));
    comm.recv_from_hp().into_bit_vec()
}

pub fn hp_sign_ge_zero<C: HpComm>(comm: &C) {
    let (msg0, msg1) = comm.recv_from_parties();
    let z0 = msg0.as_real();
    let z1 = msg1.as_real();
    let bit = if z0 + z1 >= 0.0 { 1 } else { 0 };
    comm.send_to_parties(Msg::Bit(bit), Msg::Bit(bit));
}

pub fn hp_sign_ge_zero_batch<C: HpComm>(n: usize, comm: &C) {
    let (msg0, msg1) = comm.recv_from_parties();
    let z0 = msg0.into_real_vec();
    let z1 = msg1.into_real_vec();
    assert_eq!(z0.len(), n, "p0 sign batch length mismatch");
    assert_eq!(z1.len(), n, "p1 sign batch length mismatch");
    let bits: Vec<u8> = z0
        .into_iter()
        .zip(z1)
        .map(|(a, b)| if a + b >= 0.0 { 1 } else { 0 })
        .collect();
    comm.send_to_parties(Msg::BitVec(bits.clone()), Msg::BitVec(bits));
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
    let mut acc = 0u64;
    for &val in u.iter().rev() {
        acc = add(acc, val);
        values.push(sub(acc, public_one));
    }
    values[1..].reverse();

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

pub fn party_sign_bicoptor_batch<C: PartyComm>(
    id: u8,
    shares_x: &[u64],
    lx: u32,
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<u8> {
    assert!(lx > 0 && lx < 63, "lx must be in 1..63");
    let count = (lx + 2) as usize;
    let per_item = bicoptor_randoms_per_item(count);
    let start_counter = prg0.counter();
    let mut all_values = vec![0u64; shares_x.len() * count];
    let mut ts = vec![0u8; shares_x.len()];

    let par_min = std::env::var("MCU_BICOPTOR_PAR_MIN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096);
    if shares_x.len() >= par_min {
        let prg = &*prg0;
        ts.par_iter_mut()
            .zip(all_values.par_chunks_mut(count))
            .enumerate()
            .for_each(|(idx, (t_out, out))| {
                *t_out = fill_bicoptor_values_for_share(
                    id,
                    shares_x[idx],
                    lx,
                    count,
                    start_counter + idx as u64 * per_item,
                    prg,
                    out,
                );
            });
    } else {
        for (idx, (&share_x, out)) in shares_x
            .iter()
            .zip(all_values.chunks_mut(count))
            .enumerate()
        {
            ts[idx] = fill_bicoptor_values_for_share(
                id,
                share_x,
                lx,
                count,
                start_counter + idx as u64 * per_item,
                prg0,
                out,
            );
        }
    }
    prg0.seek(start_counter + shares_x.len() as u64 * per_item);

    comm.send_to_hp(Msg::ShareVec(all_values));
    let dprimes = comm.recv_from_hp().into_bit_vec();
    assert_eq!(dprimes.len(), shares_x.len(), "HP sign-bicoptor batch length mismatch");
    dprimes
        .into_iter()
        .zip(ts)
        .map(|(dprime, t)| if t == 0 { dprime } else { 1 - dprime })
        .collect()
}

pub fn hp_sign_bicoptor<C: HpComm>(lx: u32, comm: &C) {
    let count = (lx + 2) as usize;
    let (msg0, msg1) = comm.recv_from_parties();
    let w0_values = msg0.into_share_vec();
    let w1_values = msg1.into_share_vec();
    assert_eq!(w0_values.len(), count, "p0 sign-bicoptor vector length mismatch");
    assert_eq!(w1_values.len(), count, "p1 sign-bicoptor vector length mismatch");
    let mut has_zero = false;
    for (w0, w1) in w0_values.into_iter().zip(w1_values) {
        if add(w0, w1) == 0 {
            has_zero = true;
        }
    }
    let dprime = if has_zero { 1 } else { 0 };
    comm.send_to_parties(Msg::Bit(dprime), Msg::Bit(dprime));
}

pub fn hp_sign_bicoptor_batch<C: HpComm>(n: usize, lx: u32, comm: &C) {
    let count = (lx + 2) as usize;
    let (msg0, msg1) = comm.recv_from_parties();
    let w0_values = msg0.into_share_vec();
    let w1_values = msg1.into_share_vec();
    assert_eq!(w0_values.len(), n * count, "p0 sign-bicoptor batch length mismatch");
    assert_eq!(w1_values.len(), n * count, "p1 sign-bicoptor batch length mismatch");

    let par_min = std::env::var("MCU_BICOPTOR_HP_PAR_MIN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096);
    let bits: Vec<u8> = if n >= par_min {
        w0_values
            .par_chunks(count)
            .zip(w1_values.par_chunks(count))
            .map(|(left, right)| {
                if left.iter().zip(right).any(|(&w0, &w1)| add(w0, w1) == 0) {
                    1
                } else {
                    0
                }
            })
            .collect()
    } else {
        let mut bits = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * count;
            let end = start + count;
            let has_zero = w0_values[start..end]
                .iter()
                .zip(&w1_values[start..end])
                .any(|(&w0, &w1)| add(w0, w1) == 0);
            bits.push(if has_zero { 1 } else { 0 });
        }
        bits
    };
    comm.send_to_parties(Msg::BitVec(bits.clone()), Msg::BitVec(bits));
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

pub fn party_wrap_batch<C: PartyComm>(
    id: u8,
    shares_x: &[f64],
    rs: &[f64],
    m: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<i64> {
    assert_eq!(shares_x.len(), rs.len(), "wrap batch length mismatch");
    let hi_inputs: Vec<f64> = shares_x
        .iter()
        .zip(rs)
        .map(|(&share_x, &r)| share_x - if id == 0 { m - r } else { 0.0 })
        .collect();
    let lo_inputs: Vec<f64> = shares_x
        .iter()
        .zip(rs)
        .map(|(&share_x, &r)| share_x - if id == 0 { -r } else { 0.0 })
        .collect();
    let ge_hi = party_sign_ge_zero_batch(&hi_inputs, prg0, comm);
    let ge_lo = party_sign_ge_zero_batch(&lo_inputs, prg0, comm);
    ge_hi
        .into_iter()
        .zip(ge_lo)
        .map(|(hi, lo)| hi as i64 - (1 - lo as i64))
        .collect()
}

pub fn hp_wrap<C: HpComm>(comm: &C) {
    hp_sign_ge(comm);
    hp_sign_ge(comm);
}

pub fn hp_wrap_batch<C: HpComm>(n: usize, comm: &C) {
    hp_sign_ge_zero_batch(n, comm);
    hp_sign_ge_zero_batch(n, comm);
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

pub fn party_wrap_bicoptor_batch<C: PartyComm>(
    id: u8,
    shares_x: &[f64],
    rs: &[f64],
    m: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<i64> {
    assert_eq!(shares_x.len(), rs.len(), "wrap-bicoptor batch length mismatch");
    let n = shares_x.len();
    let mut inputs = Vec::with_capacity(n * 2);
    for (&share_x, &r) in shares_x.iter().zip(rs) {
        inputs.push(fixed_compare_share(id, share_x, m - r));
        inputs.push(fixed_compare_share(id, share_x, -r));
    }
    let signs = party_sign_bicoptor_batch(id, &inputs, wrap_fixed_lx(), prg0, comm);
    assert_eq!(signs.len(), n * 2, "wrap-bicoptor sign batch length mismatch");
    signs
        .chunks_exact(2)
        .map(|pair| pair[0] as i64 - (1 - pair[1] as i64))
        .collect()
}

fn bicoptor_randoms_per_item(count: usize) -> u64 {
    // input mask + flip bit + multiplicative masks + shuffle choices + reshare masks
    (1 + 1 + count + (count - 1) + count) as u64
}

fn fill_bicoptor_values_for_share(
    id: u8,
    share_x: u64,
    lx: u32,
    count: usize,
    base_counter: u64,
    prg: &PrgSync,
    out: &mut [u64],
) -> u8 {
    debug_assert_eq!(out.len(), count);
    let input_mask = prg.raw64_at(base_counter);
    let share_x = if id == 0 {
        add(share_x, input_mask)
    } else {
        sub(share_x, input_mask)
    };
    let t = (prg.raw64_at(base_counter + 1) & 1) as u8;
    let x_share = if t == 0 { share_x } else { share_x.wrapping_neg() };

    let u_star_const = if t == 0 { 1u64 } else { 0u64.wrapping_sub(1) };
    let u_star = if id == 0 { u_star_const } else { 0 };
    let public_one = if id == 0 { 1 } else { 0 };

    out[0] = sub(add(u_star, mul(3, x_share)), public_one);
    let mut suffix = 0u64;
    for k in 0..=lx {
        suffix = add(suffix, trc_share(id, x_share, k));
    }
    for k in 0..=lx {
        let v = trc_share(id, x_share, k);
        let idx = k as usize;
        out[idx + 1] = sub(suffix, public_one);
        suffix = sub(suffix, v);
    }

    let mask_start = base_counter + 2;
    for (idx, value) in out.iter_mut().enumerate() {
        *value = mul(*value, prg.raw64_at(mask_start + idx as u64) | 1);
    }

    let shuffle_start = mask_start + count as u64;
    for i in (1..count).rev() {
        let offset = (count - 1 - i) as u64;
        let j = (prg.raw64_at(shuffle_start + offset) as usize) % (i + 1);
        out.swap(i, j);
    }

    let reshare_start = shuffle_start + (count - 1) as u64;
    for (idx, value) in out.iter_mut().enumerate() {
        let reshare_mask = prg.raw64_at(reshare_start + idx as u64);
        *value = if id == 0 {
            add(*value, reshare_mask)
        } else {
            sub(*value, reshare_mask)
        };
    }
    t
}

pub fn hp_wrap_bicoptor<C: HpComm>(comm: &C) {
    let lx = wrap_fixed_lx();
    hp_sign_bicoptor(lx, comm);
    hp_sign_bicoptor(lx, comm);
}

pub fn hp_wrap_bicoptor_batch<C: HpComm>(n: usize, comm: &C) {
    hp_sign_bicoptor_batch(n * 2, wrap_fixed_lx(), comm);
}

fn party_sign_ge_fixed<C: PartyComm>(
    id: u8,
    share_x: f64,
    threshold: f64,
    prg0: &mut PrgSync,
    comm: &C,
) -> u8 {
    let fixed_share = fixed_compare_share(id, share_x, threshold);
    party_sign_bicoptor(id, fixed_share, wrap_fixed_lx(), prg0, comm)
}

fn fixed_compare_share(id: u8, share_x: f64, threshold: f64) -> u64 {
    let shifted = share_x - if id == 0 { threshold } else { 0.0 };
    let fixed = (shifted * WRAP_FIXED_SCALE).round() as i64;
    fixed as u64
}

fn wrap_fixed_lx() -> u32 {
    let lx = std::env::var("MCU_WRAP_FIXED_LX")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(WRAP_FIXED_LX_DEFAULT);
    assert!(lx > WRAP_FIXED_SCALE_BITS && lx < 63, "MCU_WRAP_FIXED_LX must be in 25..63");
    lx
}

fn exp_par_min() -> usize {
    std::env::var("MCU_EXP_PAR_MIN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(EXP_PAR_MIN_DEFAULT)
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

fn party_exp_with_correction_batch<C: PartyComm>(
    id: u8,
    shares_x: &[f64],
    prg0: &mut PrgSync,
    comm: &C,
) -> (Vec<f64>, Vec<f64>) {
    let n = shares_x.len();
    let exp_par_min = exp_par_min();
    let start_counter = prg0.counter();
    let (rs, masked): (Vec<f64>, Vec<f64>) = if n >= exp_par_min {
        let prg = &*prg0;
        let pairs: Vec<(f64, f64)> = shares_x
            .par_iter()
            .enumerate()
            .map(|(idx, &share_x)| {
                let r = unit_from(prg.raw64_at(start_counter + idx as u64)) * MOD;
                let masked = if id == 0 {
                    (share_x + r).rem_euclid(MOD)
                } else {
                    share_x.rem_euclid(MOD)
                };
                (r, masked)
            })
            .collect();
        pairs.into_iter().unzip()
    } else {
        let mut rs = Vec::with_capacity(n);
        let mut masked = Vec::with_capacity(n);
        for &share_x in shares_x {
            let r = prg0.next_real(MOD);
            rs.push(r);
            masked.push(if id == 0 {
                (share_x + r).rem_euclid(MOD)
            } else {
                share_x.rem_euclid(MOD)
            });
        }
        (rs, masked)
    };
    if n >= exp_par_min {
        prg0.seek(start_counter + n as u64);
    }
    comm.send_to_hp(Msg::RealVec(masked));
    let shares = comm.recv_from_hp().into_real_vec();
    assert_eq!(shares.len(), n, "HP exp batch share length mismatch");
    let wraps = party_wrap_bicoptor_batch(id, shares_x, &rs, MOD, prg0, comm);
    let (corrected, corrections): (Vec<f64>, Vec<f64>) = if n >= exp_par_min {
        shares
            .par_iter()
            .zip(wraps.par_iter())
            .zip(rs.par_iter())
            .map(|((&share, &w), &r)| {
                let correction = ((w as f64) * MOD - r).exp();
                (share * correction, correction)
            })
            .unzip()
    } else {
        let corrections: Vec<f64> = wraps
            .into_iter()
            .zip(rs)
            .map(|(w, r)| ((w as f64) * MOD - r).exp())
            .collect();
        let corrected = shares
            .into_iter()
            .zip(corrections.iter())
            .map(|(share, correction)| share * correction)
            .collect();
        (corrected, corrections)
    };
    (corrected, corrections)
}

pub fn party_exp<C: PartyComm>(id: u8, share_x: f64, prg0: &mut PrgSync, comm: &C) -> f64 {
    party_exp_with_correction(id, share_x, prg0, comm).0
}

pub fn party_exp_batch<C: PartyComm>(
    id: u8,
    shares_x: &[f64],
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<f64> {
    party_exp_with_correction_batch(id, shares_x, prg0, comm).0
}

fn hp_exp_masked<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) -> f64 {
    let (msg0, msg1) = comm.recv_from_parties();
    let m0 = msg0.as_real();
    let m1 = msg1.as_real();
    let r = (m0 + m1).rem_euclid(MOD);
    let e = r.exp();
    let u = asprg_p0.next_unit();
    let s0 = u * e;
    let s1 = e - s0;
    comm.send_to_parties(Msg::Real(s0), Msg::Real(s1));
    hp_wrap_bicoptor(comm);
    e
}

fn hp_exp_masked_batch<C: HpComm>(n: usize, asprg_p0: &mut PrgSync, comm: &C) -> Vec<f64> {
    let (msg0, msg1) = comm.recv_from_parties();
    let m0 = msg0.into_real_vec();
    let m1 = msg1.into_real_vec();
    assert_eq!(m0.len(), n, "p0 exp batch length mismatch");
    assert_eq!(m1.len(), n, "p1 exp batch length mismatch");
    let exp_par_min = exp_par_min();
    let start_counter = asprg_p0.counter();
    let (masked_exps, out0, out1): (Vec<f64>, Vec<f64>, Vec<f64>) = if n >= exp_par_min {
        let prg = &*asprg_p0;
        let triples: Vec<(f64, f64, f64)> = m0
            .par_iter()
            .zip(m1.par_iter())
            .enumerate()
            .map(|(idx, (&a, &b))| {
                let r = (a + b).rem_euclid(MOD);
                let e = r.exp();
                let u = unit_from(prg.raw64_at(start_counter + idx as u64));
                let s0 = u * e;
                (e, s0, e - s0)
            })
            .collect();
        let mut masked_exps = Vec::with_capacity(n);
        let mut out0 = Vec::with_capacity(n);
        let mut out1 = Vec::with_capacity(n);
        for (e, s0, s1) in triples {
            masked_exps.push(e);
            out0.push(s0);
            out1.push(s1);
        }
        asprg_p0.seek(start_counter + n as u64);
        (masked_exps, out0, out1)
    } else {
        let mut masked_exps = Vec::with_capacity(n);
        let mut out0 = Vec::with_capacity(n);
        let mut out1 = Vec::with_capacity(n);
        for (a, b) in m0.into_iter().zip(m1) {
            let r = (a + b).rem_euclid(MOD);
            let e = r.exp();
            let u = asprg_p0.next_unit();
            out0.push(u * e);
            out1.push(e - out0.last().copied().unwrap());
            masked_exps.push(e);
        }
        (masked_exps, out0, out1)
    };
    comm.send_to_parties(Msg::RealVec(out0), Msg::RealVec(out1));
    hp_wrap_bicoptor_batch(n, comm);
    masked_exps
}

pub fn hp_exp<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) {
    hp_exp_masked(asprg_p0, comm);
}

pub fn hp_exp_batch<C: HpComm>(n: usize, asprg_p0: &mut PrgSync, comm: &C) {
    hp_exp_masked_batch(n, asprg_p0, comm);
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

pub fn party_sigmoid_batch<C: PartyComm>(
    id: u8,
    shares_z: &[f64],
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<f64> {
    let (e_shares, corrections) = party_exp_with_correction_batch(id, shares_z, prg0, comm);
    let mut ets = Vec::with_capacity(shares_z.len());
    let mut masked_denoms = Vec::with_capacity(shares_z.len());
    for e_i in e_shares {
        let d_i = e_i + if id == 0 { 1.0 } else { 0.0 };
        let t = prg0.next_real(MOD);
        let et = t.exp();
        ets.push(et);
        masked_denoms.push(et * d_i);
    }
    comm.send_to_hp(Msg::RealVec(masked_denoms));
    let fraction_shares = comm.recv_from_hp().into_real_vec();
    assert_eq!(
        fraction_shares.len(),
        shares_z.len(),
        "HP sigmoid batch share length mismatch"
    );
    fraction_shares
        .into_iter()
        .zip(ets)
        .zip(corrections)
        .map(|((share, et), correction)| share * et * correction)
        .collect()
}

pub fn hp_sigmoid<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) {
    let masked_numerator = hp_exp_masked(asprg_p0, comm);
    let (msg0, msg1) = comm.recv_from_parties();
    let u0 = msg0.as_real();
    let u1 = msg1.as_real();
    let denominator = u0 + u1;
    let fraction = masked_numerator / denominator;
    let u = asprg_p0.next_unit();
    let s0 = u * fraction;
    let s1 = fraction - s0;
    comm.send_to_parties(Msg::Real(s0), Msg::Real(s1));
}

pub fn hp_sigmoid_batch<C: HpComm>(n: usize, asprg_p0: &mut PrgSync, comm: &C) {
    let masked_numerators = hp_exp_masked_batch(n, asprg_p0, comm);
    let (msg0, msg1) = comm.recv_from_parties();
    let u0 = msg0.into_real_vec();
    let u1 = msg1.into_real_vec();
    assert_eq!(u0.len(), n, "p0 sigmoid denominator batch length mismatch");
    assert_eq!(u1.len(), n, "p1 sigmoid denominator batch length mismatch");
    let mut out0 = Vec::with_capacity(n);
    let mut out1 = Vec::with_capacity(n);
    for ((masked_numerator, a), b) in masked_numerators.into_iter().zip(u0).zip(u1) {
        let fraction = masked_numerator / (a + b);
        let u = asprg_p0.next_unit();
        out0.push(u * fraction);
        out1.push(fraction - out0.last().copied().unwrap());
    }
    comm.send_to_parties(Msg::RealVec(out0), Msg::RealVec(out1));
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

pub fn party_softmax_batch<C: PartyComm>(
    id: u8,
    shares: &[Vec<f64>],
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<Vec<f64>> {
    if shares.is_empty() {
        return Vec::new();
    }
    let k = shares[0].len();
    assert!(shares.iter().all(|row| row.len() == k), "softmax batch ragged input");
    let flat: Vec<f64> = shares.iter().flatten().copied().collect();
    let (exp_shares, corrections) = party_exp_with_correction_batch(id, &flat, prg0, comm);
    let mut ets = Vec::with_capacity(shares.len());
    let mut masked_denoms = Vec::with_capacity(shares.len());
    for row in exp_shares.chunks_exact(k) {
        let t = prg0.next_real(MOD);
        let et = t.exp();
        ets.push(et);
        masked_denoms.push(et * row.iter().sum::<f64>());
    }
    comm.send_to_hp(Msg::RealVec(masked_denoms));
    let fraction_shares = comm.recv_from_hp().into_real_vec();
    assert_eq!(
        fraction_shares.len(),
        flat.len(),
        "HP softmax batch share length mismatch"
    );
    let mut out = Vec::with_capacity(shares.len());
    for (row_idx, row) in fraction_shares.chunks_exact(k).enumerate() {
        let mut out_row = Vec::with_capacity(k);
        for j in 0..k {
            let idx = row_idx * k + j;
            out_row.push(row[j] * ets[row_idx] * corrections[idx]);
        }
        out.push(out_row);
    }
    out
}

pub fn hp_softmax<C: HpComm>(k: usize, asprg_p0: &mut PrgSync, comm: &C) {
    let mut masked_numerators = Vec::with_capacity(k);
    for _ in 0..k {
        masked_numerators.push(hp_exp_masked(asprg_p0, comm));
    }
    let (msg0, msg1) = comm.recv_from_parties();
    let u0 = msg0.as_real();
    let u1 = msg1.as_real();
    let denominator = u0 + u1;
    for masked_numerator in masked_numerators {
        let fraction = masked_numerator / denominator;
        let u = asprg_p0.next_unit();
        let s0 = u * fraction;
        let s1 = fraction - s0;
        comm.send_to_parties(Msg::Real(s0), Msg::Real(s1));
    }
}

pub fn hp_softmax_batch<C: HpComm>(n: usize, k: usize, asprg_p0: &mut PrgSync, comm: &C) {
    let masked_numerators = hp_exp_masked_batch(n * k, asprg_p0, comm);
    let (msg0, msg1) = comm.recv_from_parties();
    let u0 = msg0.into_real_vec();
    let u1 = msg1.into_real_vec();
    assert_eq!(u0.len(), n, "p0 softmax denominator batch length mismatch");
    assert_eq!(u1.len(), n, "p1 softmax denominator batch length mismatch");
    let mut out0 = Vec::with_capacity(n * k);
    let mut out1 = Vec::with_capacity(n * k);
    for row in 0..n {
        let denominator = u0[row] + u1[row];
        for masked_numerator in &masked_numerators[row * k..(row + 1) * k] {
            let fraction = masked_numerator / denominator;
            let u = asprg_p0.next_unit();
            out0.push(u * fraction);
            out1.push(fraction - out0.last().copied().unwrap());
        }
    }
    comm.send_to_parties(Msg::RealVec(out0), Msg::RealVec(out1));
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

pub fn party_gelu_batch<C: PartyComm>(
    id: u8,
    shares_x: &[f64],
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<f64> {
    let scaled: Vec<f64> = shares_x.iter().map(|&share_x| GELU_COEF * share_x).collect();
    let sig_shares = party_sigmoid_batch(id, &scaled, prg0, comm);
    let mut a = Vec::with_capacity(shares_x.len());
    let mut b = Vec::with_capacity(shares_x.len());
    let mut corrections = Vec::with_capacity(shares_x.len());
    for (&share_x, &sig_i) in shares_x.iter().zip(&sig_shares) {
        let r_a = prg0.next_real(MOD);
        let r_b = prg0.next_real(MOD);
        if id == 0 {
            a.push(share_x + r_a);
            b.push(sig_i + r_b);
            corrections.push(share_x * r_b + sig_i * r_a + r_a * r_b);
        } else {
            a.push(share_x);
            b.push(sig_i);
            corrections.push(share_x * r_b + sig_i * r_a);
        }
    }
    comm.send_to_hp(Msg::RealPairVec { a, b });
    let hp_shares = comm.recv_from_hp().into_real_vec();
    assert_eq!(hp_shares.len(), shares_x.len(), "HP gelu batch share length mismatch");
    hp_shares
        .into_iter()
        .zip(corrections)
        .map(|(share, correction)| share - correction)
        .collect()
}

pub fn hp_gelu<C: HpComm>(asprg_p0: &mut PrgSync, comm: &C) {
    hp_sigmoid(asprg_p0, comm);
    let (msg0, msg1) = comm.recv_from_parties();
    let (a0, b0) = msg0.as_real_pair();
    let (a1, b1) = msg1.as_real_pair();
    let product = (a0 + a1) * (b0 + b1);
    let u = asprg_p0.next_unit();
    let s0 = u * product;
    let s1 = product - s0;
    comm.send_to_parties(Msg::Real(s0), Msg::Real(s1));
}

pub fn hp_gelu_batch<C: HpComm>(n: usize, asprg_p0: &mut PrgSync, comm: &C) {
    hp_sigmoid_batch(n, asprg_p0, comm);
    let (msg0, msg1) = comm.recv_from_parties();
    let (a0, b0) = msg0.into_real_pair_vec();
    let (a1, b1) = msg1.into_real_pair_vec();
    assert_eq!(a0.len(), n, "p0 gelu batch a length mismatch");
    assert_eq!(a1.len(), n, "p1 gelu batch a length mismatch");
    assert_eq!(b0.len(), n, "p0 gelu batch b length mismatch");
    assert_eq!(b1.len(), n, "p1 gelu batch b length mismatch");
    let mut out0 = Vec::with_capacity(n);
    let mut out1 = Vec::with_capacity(n);
    for (((a0, b0), a1), b1) in a0.into_iter().zip(b0).zip(a1).zip(b1) {
        let product = (a0 + a1) * (b0 + b1);
        let u = asprg_p0.next_unit();
        out0.push(u * product);
        out1.push(product - out0.last().copied().unwrap());
    }
    comm.send_to_parties(Msg::RealVec(out0), Msg::RealVec(out1));
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

    fn run_vec<F0, F1, FH>(f0: F0, f1: F1, fh: FH) -> (Vec<f64>, Vec<f64>)
    where
        F0: FnOnce(crate::channel::PartyEndpoint) -> Vec<f64> + Send + 'static,
        F1: FnOnce(crate::channel::PartyEndpoint) -> Vec<f64> + Send + 'static,
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
    fn real_exp_batch_comm_matches_plain() {
        let xs = vec![-1.5, 0.0, 1.25, 2.0];
        let xs0 = vec![7.0, -4.0, 0.25, 2.0];
        let xs1: Vec<f64> = xs.iter().zip(xs0.iter()).map(|(x, x0)| x - x0).collect();
        let n = xs.len();
        let (o0, o1) = run_vec(
            move |comm| party_exp_batch(0, &xs0, &mut PrgSync::new(&S0), &comm),
            move |comm| party_exp_batch(1, &xs1, &mut PrgSync::new(&S0), &comm),
            move |comm| hp_exp_batch(n, &mut PrgSync::new(&S1), &comm),
        );
        for i in 0..xs.len() {
            assert!((o0[i] + o1[i] - xs[i].exp()).abs() < 1e-10, "i={i}");
        }
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
    fn real_bicoptor_sign_batch_comm_matches_plain() {
        let lx = 16;
        let xs = vec![1234i64, -5678i64, 1i64, -1i64, 4096i64];
        let x0: Vec<u64> = (0..xs.len())
            .map(|i| 0x1234_5678_ABCD_EF01u64.wrapping_add(i as u64))
            .collect();
        let x1: Vec<u64> = xs
            .iter()
            .zip(x0.iter())
            .map(|(&x, &s0)| (x as u64).wrapping_sub(s0))
            .collect();
        let n = xs.len();
        let (p0, p1, hp) = make_mock();
        let t0 = thread::spawn(move || {
            party_sign_bicoptor_batch(0, &x0, lx, &mut PrgSync::new(&S0), &p0)
        });
        let t1 = thread::spawn(move || {
            party_sign_bicoptor_batch(1, &x1, lx, &mut PrgSync::new(&S0), &p1)
        });
        let th = thread::spawn(move || hp_sign_bicoptor_batch(n, lx, &hp));
        let o0 = t0.join().unwrap();
        let o1 = t1.join().unwrap();
        th.join().unwrap();
        for i in 0..xs.len() {
            let want = if xs[i] >= 0 { 1 } else { 0 };
            assert_eq!(o0[i], want, "p0 i={i}");
            assert_eq!(o1[i], want, "p1 i={i}");
        }
    }

    #[test]
    fn real_bicoptor_sign_comm_matches_fixed_wrap_input() {
        let lx = wrap_fixed_lx();
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
        let lx = wrap_fixed_lx();
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
    fn real_sigmoid_batch_comm_matches_plain() {
        let xs = vec![-2.0, -0.5, 0.75, 3.0];
        let xs0 = vec![3.0, -1.0, 4.0, 0.25];
        let xs1: Vec<f64> = xs.iter().zip(xs0.iter()).map(|(x, x0)| x - x0).collect();
        let n = xs.len();
        let (o0, o1) = run_vec(
            move |comm| party_sigmoid_batch(0, &xs0, &mut PrgSync::new(&S0), &comm),
            move |comm| party_sigmoid_batch(1, &xs1, &mut PrgSync::new(&S0), &comm),
            move |comm| hp_sigmoid_batch(n, &mut PrgSync::new(&S1), &comm),
        );
        for i in 0..xs.len() {
            let want = 1.0 / (1.0 + (-xs[i]).exp());
            assert!((o0[i] + o1[i] - want).abs() < 1e-10, "i={i}");
        }
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
    fn real_gelu_batch_comm_matches_plain() {
        let xs = vec![-2.0, 0.0, 0.75, 2.5];
        let xs0 = vec![-4.0, 1.0, -0.25, 3.0];
        let xs1: Vec<f64> = xs.iter().zip(xs0.iter()).map(|(x, x0)| x - x0).collect();
        let n = xs.len();
        let (o0, o1) = run_vec(
            move |comm| party_gelu_batch(0, &xs0, &mut PrgSync::new(&S0), &comm),
            move |comm| party_gelu_batch(1, &xs1, &mut PrgSync::new(&S0), &comm),
            move |comm| hp_gelu_batch(n, &mut PrgSync::new(&S1), &comm),
        );
        for i in 0..xs.len() {
            let want = xs[i] * (1.0 / (1.0 + (-(GELU_COEF * xs[i])).exp()));
            assert!((o0[i] + o1[i] - want).abs() < 1e-10, "i={i}");
        }
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

    #[test]
    fn real_softmax_batch_comm_matches_plain() {
        let xs = vec![
            vec![1.0, -2.0, 3.0, 0.5],
            vec![-1.0, 0.25, 0.75, 2.0],
            vec![3.0, 2.0, 1.0, -3.0],
        ];
        let xs0 = vec![
            vec![7.0, -4.0, 0.25, 2.0],
            vec![1.5, -2.0, 4.0, 0.0],
            vec![-1.0, 5.0, 2.5, -4.0],
        ];
        let xs1: Vec<Vec<f64>> = xs
            .iter()
            .zip(xs0.iter())
            .map(|(row, row0)| row.iter().zip(row0).map(|(x, x0)| x - x0).collect())
            .collect();
        let n = xs.len();
        let k = xs[0].len();
        let (o0, o1) = run_vec(
            move |comm| {
                party_softmax_batch(0, &xs0, &mut PrgSync::new(&S0), &comm)
                    .into_iter()
                    .flatten()
                    .collect()
            },
            move |comm| {
                party_softmax_batch(1, &xs1, &mut PrgSync::new(&S0), &comm)
                    .into_iter()
                    .flatten()
                    .collect()
            },
            move |comm| hp_softmax_batch(n, k, &mut PrgSync::new(&S1), &comm),
        );
        for row in 0..n {
            let denom: f64 = xs[row].iter().map(|x| f64::exp(*x)).sum();
            for col in 0..k {
                let idx = row * k + col;
                let want = xs[row][col].exp() / denom;
                assert!((o0[idx] + o1[idx] - want).abs() < 1e-10, "idx={idx}");
            }
        }
    }
}
