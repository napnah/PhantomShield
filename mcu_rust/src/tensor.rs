use crate::channel::{HpComm, Msg, PartyComm};
use crate::prg::PrgSync;
use crate::protocols::multiply::{hp_multiply_batch, party_multiply_batch};
use crate::ring::{add, mul};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[cfg(feature = "cuda")]
extern "C" {
    fn mcu_cuda_matmul_u64(
        a_host: *const u64,
        b_host: *const u64,
        out_host: *mut u64,
        m: usize,
        k: usize,
        n: usize,
    ) -> i32;
    fn mcu_cuda_party_matmul_finish_u64(
        a_host: *const u64,
        b_host: *const u64,
        ra_host: *const u64,
        rb_host: *const u64,
        hp_share_host: *const u64,
        out_host: *mut u64,
        m: usize,
        k: usize,
        n: usize,
        include_rarb: i32,
    ) -> i32;
    fn mcu_cuda_hp_matmul_share_u64(
        a0_host: *const u64,
        a1_host: *const u64,
        b0_host: *const u64,
        b1_host: *const u64,
        out0_host: *const u64,
        out1_host: *mut u64,
        m: usize,
        k: usize,
        n: usize,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TensorStats {
    pub matmul_calls: u64,
    pub fused_party_calls: u64,
    pub fused_hp_calls: u64,
    pub cpu_matmul_calls: u64,
    pub cuda_matmul_calls: u64,
    pub cuda_fallbacks: u64,
    pub local_nanos: u64,
}

static MATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
static FUSED_PARTY_CALLS: AtomicU64 = AtomicU64::new(0);
static FUSED_HP_CALLS: AtomicU64 = AtomicU64::new(0);
static CPU_MATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
static CUDA_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static LOCAL_NANOS: AtomicU64 = AtomicU64::new(0);

pub fn reset_tensor_stats() {
    MATMUL_CALLS.store(0, Ordering::Relaxed);
    FUSED_PARTY_CALLS.store(0, Ordering::Relaxed);
    FUSED_HP_CALLS.store(0, Ordering::Relaxed);
    CPU_MATMUL_CALLS.store(0, Ordering::Relaxed);
    CUDA_MATMUL_CALLS.store(0, Ordering::Relaxed);
    CUDA_FALLBACKS.store(0, Ordering::Relaxed);
    LOCAL_NANOS.store(0, Ordering::Relaxed);
}

pub fn tensor_stats_snapshot() -> TensorStats {
    TensorStats {
        matmul_calls: MATMUL_CALLS.load(Ordering::Relaxed),
        fused_party_calls: FUSED_PARTY_CALLS.load(Ordering::Relaxed),
        fused_hp_calls: FUSED_HP_CALLS.load(Ordering::Relaxed),
        cpu_matmul_calls: CPU_MATMUL_CALLS.load(Ordering::Relaxed),
        cuda_matmul_calls: CUDA_MATMUL_CALLS.load(Ordering::Relaxed),
        cuda_fallbacks: CUDA_FALLBACKS.load(Ordering::Relaxed),
        local_nanos: LOCAL_NANOS.load(Ordering::Relaxed),
    }
}

fn record_local(start: Instant) {
    LOCAL_NANOS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

fn cpu_parallel_min_ops() -> usize {
    std::env::var("MCU_CPU_PAR_MIN_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 20)
}

pub fn party_elemul<C: PartyComm>(
    id: u8,
    x: &[u64],
    y: &[u64],
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<u64> {
    assert_eq!(x.len(), y.len(), "elementwise multiply length mismatch");
    party_multiply_batch(id, x, y, prg0, comm)
}

pub fn hp_elemul<C: HpComm>(n: usize, asprg_p0: &mut PrgSync, comm: &C) {
    hp_multiply_batch(n, asprg_p0, comm);
}

pub fn party_matmul<C: PartyComm>(
    id: u8,
    a: &[u64],
    b: &[u64],
    m: usize,
    k: usize,
    n: usize,
    prg0: &mut PrgSync,
    comm: &C,
) -> Vec<u64> {
    assert_eq!(a.len(), m * k, "left matrix shape mismatch");
    assert_eq!(b.len(), k * n, "right matrix shape mismatch");
    let r_a: Vec<u64> = (0..m * k).map(|_| prg0.next()).collect();
    let r_b: Vec<u64> = (0..k * n).map(|_| prg0.next()).collect();

    if id == 0 {
        let masked_a: Vec<u64> = a
            .iter()
            .zip(r_a.iter())
            .map(|(&x, &r)| add(x, r))
            .collect();
        let masked_b: Vec<u64> = b
            .iter()
            .zip(r_b.iter())
            .map(|(&x, &r)| add(x, r))
            .collect();
        comm.send_to_hp(Msg::MatMulToHp {
            id,
            a: masked_a,
            b: masked_b,
        });
    } else {
        comm.send_to_hp(Msg::MatMulToHp {
            id,
            a: a.to_vec(),
            b: b.to_vec(),
        });
    }

    let hp_share = comm.recv_from_hp().into_share_vec();
    assert_eq!(hp_share.len(), m * n, "HP matmul share shape mismatch");
    party_matmul_finish(id, a, b, &r_a, &r_b, &hp_share, m, k, n)
}

pub fn hp_matmul<C: HpComm>(
    m: usize,
    k: usize,
    n: usize,
    asprg_p0: &mut PrgSync,
    comm: &C,
) {
    let (msg0, msg1) = comm.recv_from_parties();
    let (_, a0, b0) = msg0.as_matmul();
    let (_, a1, b1) = msg1.as_matmul();
    assert_eq!(a0.len(), m * k, "p0 left matrix shape mismatch");
    assert_eq!(a1.len(), m * k, "p1 left matrix shape mismatch");
    assert_eq!(b0.len(), k * n, "p0 right matrix shape mismatch");
    assert_eq!(b1.len(), k * n, "p1 right matrix shape mismatch");
    let out0: Vec<u64> = (0..m * n).map(|_| asprg_p0.next()).collect();
    let out1 = hp_matmul_share(a0, a1, b0, b1, &out0, m, k, n);
    comm.send_to_parties(Msg::ShareVec(out0), Msg::ShareVec(out1));
}

fn add_vec_untracked(a: &[u64], b: &[u64]) -> Vec<u64> {
    assert_eq!(a.len(), b.len(), "vector add length mismatch");
    if a.len() < cpu_parallel_min_ops() {
        a.iter().zip(b.iter()).map(|(&x, &y)| add(x, y)).collect()
    } else {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| x.wrapping_add(y))
            .collect()
    }
}

#[allow(dead_code)]
pub(crate) fn matmul_plain(a: &[u64], b: &[u64], m: usize, k: usize, n: usize) -> Vec<u64> {
    assert_eq!(a.len(), m * k, "left matrix shape mismatch");
    assert_eq!(b.len(), k * n, "right matrix shape mismatch");
    let start = Instant::now();
    let mut out = vec![0u64; m * n];
    if m * k * n < cpu_parallel_min_ops() {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0u64;
                for t in 0..k {
                    acc = add(acc, mul(a[i * k + t], b[t * n + j]));
                }
                out[i * n + j] = acc;
            }
        }
    } else {
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut acc = 0u64;
                for t in 0..k {
                    acc = acc.wrapping_add(a[i * k + t].wrapping_mul(b[t * n + j]));
                }
                row[j] = acc;
            }
        });
    }
    CPU_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
    record_local(start);
    out
}

#[allow(dead_code)]
fn matmul_accel(a: &[u64], b: &[u64], m: usize, k: usize, n: usize) -> Vec<u64> {
    MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
    assert_eq!(a.len(), m * k, "left matrix shape mismatch");
    assert_eq!(b.len(), k * n, "right matrix shape mismatch");
    #[cfg(feature = "cuda")]
    {
        if let Some(out) = matmul_cuda(a, b, m, k, n) {
            return out;
        }
    }
    matmul_plain(a, b, m, k, n)
}

fn party_matmul_finish(
    id: u8,
    a: &[u64],
    b: &[u64],
    r_a: &[u64],
    r_b: &[u64],
    hp_share: &[u64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u64> {
    assert_eq!(a.len(), m * k, "left matrix shape mismatch");
    assert_eq!(b.len(), k * n, "right matrix shape mismatch");
    assert_eq!(r_a.len(), m * k, "left mask shape mismatch");
    assert_eq!(r_b.len(), k * n, "right mask shape mismatch");
    assert_eq!(hp_share.len(), m * n, "HP share shape mismatch");
    #[cfg(feature = "cuda")]
    {
        if let Some(out) = party_matmul_finish_cuda(id, a, b, r_a, r_b, hp_share, m, k, n) {
            return out;
        }
    }
    party_matmul_finish_cpu(id, a, b, r_a, r_b, hp_share, m, k, n)
}

#[allow(clippy::too_many_arguments)]
fn party_matmul_finish_cpu(
    id: u8,
    a: &[u64],
    b: &[u64],
    r_a: &[u64],
    r_b: &[u64],
    hp_share: &[u64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u64> {
    let start = Instant::now();
    let include_rarb = id == 0;
    let mut out = vec![0u64; m * n];
    if m * k * n < cpu_parallel_min_ops() {
        for i in 0..m {
            for j in 0..n {
                let mut corr = 0u64;
                for t in 0..k {
                    let av = a[i * k + t];
                    let bv = b[t * n + j];
                    let rav = r_a[i * k + t];
                    let rbv = r_b[t * n + j];
                    corr = corr.wrapping_add(av.wrapping_mul(rbv));
                    corr = corr.wrapping_add(rav.wrapping_mul(bv));
                    if include_rarb {
                        corr = corr.wrapping_add(rav.wrapping_mul(rbv));
                    }
                }
                let idx = i * n + j;
                out[idx] = hp_share[idx].wrapping_sub(corr);
            }
        }
    } else {
        let b_t = transpose_kn_to_nk(b, k, n);
        let rb_t = transpose_kn_to_nk(r_b, k, n);
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut corr = 0u64;
                for t in 0..k {
                    let av = a[i * k + t];
                    let bv = b_t[j * k + t];
                    let rav = r_a[i * k + t];
                    let rbv = rb_t[j * k + t];
                    corr = corr.wrapping_add(av.wrapping_mul(rbv));
                    corr = corr.wrapping_add(rav.wrapping_mul(bv));
                    if include_rarb {
                        corr = corr.wrapping_add(rav.wrapping_mul(rbv));
                    }
                }
                row[j] = hp_share[i * n + j].wrapping_sub(corr);
            }
        });
    }
    FUSED_PARTY_CALLS.fetch_add(1, Ordering::Relaxed);
    CPU_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
    record_local(start);
    out
}

fn hp_matmul_share(
    a0: &[u64],
    a1: &[u64],
    b0: &[u64],
    b1: &[u64],
    out0: &[u64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u64> {
    assert_eq!(a0.len(), m * k, "p0 left matrix shape mismatch");
    assert_eq!(a1.len(), m * k, "p1 left matrix shape mismatch");
    assert_eq!(b0.len(), k * n, "p0 right matrix shape mismatch");
    assert_eq!(b1.len(), k * n, "p1 right matrix shape mismatch");
    assert_eq!(out0.len(), m * n, "HP output share shape mismatch");
    #[cfg(feature = "cuda")]
    {
        if let Some(out) = hp_matmul_share_cuda(a0, a1, b0, b1, out0, m, k, n) {
            return out;
        }
    }
    hp_matmul_share_cpu(a0, a1, b0, b1, out0, m, k, n)
}

#[allow(clippy::too_many_arguments)]
fn hp_matmul_share_cpu(
    a0: &[u64],
    a1: &[u64],
    b0: &[u64],
    b1: &[u64],
    out0: &[u64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u64> {
    let start = Instant::now();
    let a = add_vec_untracked(a0, a1);
    let b = add_vec_untracked(b0, b1);
    let product = matmul_plain_untracked(&a, &b, m, k, n);
    let out1 = if product.len() < cpu_parallel_min_ops() {
        product
            .iter()
            .zip(out0.iter())
            .map(|(&p, &s0)| p.wrapping_sub(s0))
            .collect()
    } else {
        product
            .par_iter()
            .zip(out0.par_iter())
            .map(|(&p, &s0)| p.wrapping_sub(s0))
            .collect()
    };
    FUSED_HP_CALLS.fetch_add(1, Ordering::Relaxed);
    CPU_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
    record_local(start);
    out1
}

fn matmul_plain_untracked(a: &[u64], b: &[u64], m: usize, k: usize, n: usize) -> Vec<u64> {
    assert_eq!(a.len(), m * k, "left matrix shape mismatch");
    assert_eq!(b.len(), k * n, "right matrix shape mismatch");
    let mut out = vec![0u64; m * n];
    if m * k * n < cpu_parallel_min_ops() {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0u64;
                for t in 0..k {
                    acc = acc.wrapping_add(a[i * k + t].wrapping_mul(b[t * n + j]));
                }
                out[i * n + j] = acc;
            }
        }
    } else {
        let b_t = transpose_kn_to_nk(b, k, n);
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut acc = 0u64;
                for t in 0..k {
                    acc = acc.wrapping_add(a[i * k + t].wrapping_mul(b_t[j * k + t]));
                }
                row[j] = acc;
            }
        });
    }
    out
}

fn transpose_kn_to_nk(values: &[u64], k: usize, n: usize) -> Vec<u64> {
    assert_eq!(values.len(), k * n, "transpose shape mismatch");
    let mut out = vec![0u64; values.len()];
    out.par_chunks_mut(k).enumerate().for_each(|(j, row)| {
        for t in 0..k {
            row[t] = values[t * n + j];
        }
    });
    out
}

#[cfg(feature = "cuda")]
fn matmul_cuda(a: &[u64], b: &[u64], m: usize, k: usize, n: usize) -> Option<Vec<u64>> {
    if m == 0 || k == 0 || n == 0 {
        return Some(vec![0u64; m * n]);
    }
    if m * k * n < cuda_min_ops() {
        CUDA_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return None;
    }

    let start = Instant::now();
    let mut out = vec![0u64; m * n];
    let code = unsafe { mcu_cuda_matmul_u64(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n) };
    if code == 0 {
        CUDA_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
        record_local(start);
        Some(out)
    } else {
        eprintln!("[mcu_cuda] matmul failed with cuda error {code}; falling back to CPU");
        CUDA_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        None
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn party_matmul_finish_cuda(
    id: u8,
    a: &[u64],
    b: &[u64],
    r_a: &[u64],
    r_b: &[u64],
    hp_share: &[u64],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<u64>> {
    if m == 0 || k == 0 || n == 0 {
        return Some(vec![0u64; m * n]);
    }
    if m * k * n < cuda_min_ops() {
        CUDA_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let start = Instant::now();
    let mut out = vec![0u64; m * n];
    let code = unsafe {
        mcu_cuda_party_matmul_finish_u64(
            a.as_ptr(),
            b.as_ptr(),
            r_a.as_ptr(),
            r_b.as_ptr(),
            hp_share.as_ptr(),
            out.as_mut_ptr(),
            m,
            k,
            n,
            if id == 0 { 1 } else { 0 },
        )
    };
    if code == 0 {
        FUSED_PARTY_CALLS.fetch_add(1, Ordering::Relaxed);
        CUDA_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
        record_local(start);
        Some(out)
    } else {
        eprintln!("[mcu_cuda] fused party matmul failed with cuda error {code}; falling back to CPU");
        CUDA_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        None
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn hp_matmul_share_cuda(
    a0: &[u64],
    a1: &[u64],
    b0: &[u64],
    b1: &[u64],
    out0: &[u64],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<u64>> {
    if m == 0 || k == 0 || n == 0 {
        return Some(vec![0u64; m * n]);
    }
    if m * k * n < cuda_min_ops() {
        CUDA_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let start = Instant::now();
    let mut out1 = vec![0u64; m * n];
    let code = unsafe {
        mcu_cuda_hp_matmul_share_u64(
            a0.as_ptr(),
            a1.as_ptr(),
            b0.as_ptr(),
            b1.as_ptr(),
            out0.as_ptr(),
            out1.as_mut_ptr(),
            m,
            k,
            n,
        )
    };
    if code == 0 {
        FUSED_HP_CALLS.fetch_add(1, Ordering::Relaxed);
        CUDA_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
        record_local(start);
        Some(out1)
    } else {
        eprintln!("[mcu_cuda] fused HP matmul failed with cuda error {code}; falling back to CPU");
        CUDA_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        None
    }
}

#[cfg(feature = "cuda")]
fn cuda_min_ops() -> usize {
    std::env::var("MCU_CUDA_MIN_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 20)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::make_mock;
    use crate::ring::sub;
    use std::thread;

    const SEED_SHARED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    const SEED_HP: [u8; 16] = [
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];

    #[test]
    fn tensor_level_matmul_comm_matches_plaintext() {
        let (m, k, n) = (2, 3, 2);
        let a = vec![1u64, 2, 3, 4, 5, 6];
        let b = vec![7u64, 8, 9, 10, 11, 12];
        let a0 = vec![101u64, 102, 103, 104, 105, 106];
        let b0 = vec![201u64, 202, 203, 204, 205, 206];
        let a1: Vec<u64> = a.iter().zip(a0.iter()).map(|(&v, &s)| sub(v, s)).collect();
        let b1: Vec<u64> = b.iter().zip(b0.iter()).map(|(&v, &s)| sub(v, s)).collect();
        let want = matmul_plain(&a, &b, m, k, n);
        let (p0, p1, hp) = make_mock();
        let t0 = thread::spawn(move || {
            party_matmul(
                0,
                &a0,
                &b0,
                m,
                k,
                n,
                &mut PrgSync::new(&SEED_SHARED),
                &p0,
            )
        });
        let t1 = thread::spawn(move || {
            party_matmul(
                1,
                &a1,
                &b1,
                m,
                k,
                n,
                &mut PrgSync::new(&SEED_SHARED),
                &p1,
            )
        });
        let th = thread::spawn(move || hp_matmul(m, k, n, &mut PrgSync::new(&SEED_HP), &hp));
        let o0 = t0.join().unwrap();
        let o1 = t1.join().unwrap();
        th.join().unwrap();
        for i in 0..want.len() {
            assert_eq!(add(o0[i], o1[i]), want[i], "i={i}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matmul_matches_plaintext_when_forced() {
        std::env::set_var("MCU_CUDA_MIN_OPS", "1");
        let (m, k, n) = (16, 16, 16);
        let a: Vec<u64> = (0..m * k).map(|i| i as u64 * 3 + 1).collect();
        let b: Vec<u64> = (0..k * n).map(|i| i as u64 * 5 + 7).collect();
        let got = matmul_accel(&a, &b, m, k, n);
        let want = matmul_plain(&a, &b, m, k, n);
        assert_eq!(got, want);
        std::env::remove_var("MCU_CUDA_MIN_OPS");
    }
}
