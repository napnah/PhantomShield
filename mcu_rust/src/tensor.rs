use crate::channel::{HpComm, Msg, PartyComm};
use crate::prg::PrgSync;
use crate::protocols::multiply::{hp_multiply_batch, party_multiply_batch};
use crate::ring::{add, mul, sub};

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
    let correction = if id == 0 {
        let arb = matmul_plain(a, &r_b, m, k, n);
        let rab = matmul_plain(&r_a, b, m, k, n);
        let rarb = matmul_plain(&r_a, &r_b, m, k, n);
        add_vec(&add_vec(&arb, &rab), &rarb)
    } else {
        let arb = matmul_plain(a, &r_b, m, k, n);
        let rab = matmul_plain(&r_a, b, m, k, n);
        add_vec(&arb, &rab)
    };
    hp_share
        .iter()
        .zip(correction.iter())
        .map(|(&share, &corr)| sub(share, corr))
        .collect()
}

pub fn hp_matmul<C: HpComm>(
    m: usize,
    k: usize,
    n: usize,
    asprg_p0: &mut PrgSync,
    comm: &C,
) {
    let msg0 = comm.recv_from_p0();
    let msg1 = comm.recv_from_p1();
    let (_, a0, b0) = msg0.as_matmul();
    let (_, a1, b1) = msg1.as_matmul();
    assert_eq!(a0.len(), m * k, "p0 left matrix shape mismatch");
    assert_eq!(a1.len(), m * k, "p1 left matrix shape mismatch");
    assert_eq!(b0.len(), k * n, "p0 right matrix shape mismatch");
    assert_eq!(b1.len(), k * n, "p1 right matrix shape mismatch");
    let a = add_vec(a0, a1);
    let b = add_vec(b0, b1);
    let product = matmul_plain(&a, &b, m, k, n);
    let out0: Vec<u64> = (0..m * n).map(|_| asprg_p0.next()).collect();
    let out1: Vec<u64> = product
        .iter()
        .zip(out0.iter())
        .map(|(&p, &s0)| sub(p, s0))
        .collect();
    comm.send_to_p0(Msg::ShareVec(out0));
    comm.send_to_p1(Msg::ShareVec(out1));
}

fn add_vec(a: &[u64], b: &[u64]) -> Vec<u64> {
    assert_eq!(a.len(), b.len(), "vector add length mismatch");
    a.iter().zip(b.iter()).map(|(&x, &y)| add(x, y)).collect()
}

fn matmul_plain(a: &[u64], b: &[u64], m: usize, k: usize, n: usize) -> Vec<u64> {
    assert_eq!(a.len(), m * k, "left matrix shape mismatch");
    assert_eq!(b.len(), k * n, "right matrix shape mismatch");
    let mut out = vec![0u64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0u64;
            for t in 0..k {
                acc = add(acc, mul(a[i * k + t], b[t * n + j]));
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::make_mock;
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
}
