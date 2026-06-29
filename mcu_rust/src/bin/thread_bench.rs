use std::env;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use mcu_rust::channel::{make_mock, HpEndpoint, PartyEndpoint};
use mcu_rust::prg::PrgSync;
use mcu_rust::protocols::GELU_COEF;
use mcu_rust::real_protocols::{
    hp_exp_batch, hp_gelu_batch, hp_sigmoid_batch, hp_softmax_batch, party_exp_batch,
    party_gelu_batch, party_sigmoid_batch, party_softmax_batch,
};
use mcu_rust::ring::{add, mul, sub};
use mcu_rust::tensor::{hp_elemul, hp_matmul, party_elemul, party_matmul};

const SHARED_SEED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const HP_P0_SEED: [u8; 16] = [
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Elemul,
    Matmul,
    Exp,
    Sigmoid,
    Softmax,
    Gelu,
}

#[derive(Debug)]
enum NonlinearOut {
    Reals(Vec<f64>),
    Rows(Vec<Vec<f64>>),
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(state: &mut u64) -> f64 {
    const DEN: f64 = (1u64 << 53) as f64;
    (splitmix64(state) % (1u64 << 53)) as f64 / DEN
}

fn make_vec(seed: u64, len: usize) -> Vec<u64> {
    let mut s = seed;
    (0..len).map(|_| splitmix64(&mut s)).collect()
}

fn share_vec(values: &[u64], seed: u64, id: u8) -> Vec<u64> {
    let shares0 = make_vec(seed, values.len());
    values
        .iter()
        .zip(shares0.iter())
        .map(|(&v, &s0)| if id == 0 { s0 } else { sub(v, s0) })
        .collect()
}

fn value(i: usize, op: Op) -> f64 {
    let mut s = 0xCAFE_BABE_D15E_A5E5u64 ^ ((i as u64) << 7);
    match op {
        Op::Exp | Op::Sigmoid => -8.0 + unit(&mut s) * 16.0,
        Op::Gelu => -5.0 + unit(&mut s) * 10.0,
        _ => unreachable!("tensor and softmax ops use dedicated inputs"),
    }
}

fn vector_value(i: usize, k: usize) -> Vec<f64> {
    let mut s = 0x534F_F74D_AA55_1001u64 ^ ((i as u64) << 9);
    (0..k).map(|_| -8.0 + unit(&mut s) * 16.0).collect()
}

fn share_value(v: f64, i: usize, id: u8) -> f64 {
    let mut s = 0x5123_4567_89AB_CDEFu64 ^ ((i as u64) << 5);
    let s0 = -50.0 + unit(&mut s) * 100.0;
    if id == 0 { s0 } else { v - s0 }
}

fn elemul_inputs(len: usize) -> (Vec<u64>, Vec<u64>) {
    (make_vec(0xA001, len), make_vec(0xB002, len))
}

fn matmul_inputs(m: usize, k: usize, n: usize) -> (Vec<u64>, Vec<u64>) {
    (make_vec(0xC003, m * k), make_vec(0xD004, k * n))
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  thread_bench --op elemul --len 10000\n  thread_bench --op matmul --m 16 --k 64 --n 16\n  thread_bench --op exp --n 100\n  thread_bench --op sigmoid --n 100\n  thread_bench --op gelu --n 100\n  thread_bench --op softmax --n 16 --k 4"
    );
    std::process::exit(2);
}

fn arg_value(args: &[String], name: &str, default: Option<&str>) -> String {
    for pair in args.windows(2) {
        if pair[0] == name {
            return pair[1].clone();
        }
    }
    default.map(str::to_string).unwrap_or_else(|| usage())
}

fn parse_op(args: &[String]) -> Op {
    match arg_value(args, "--op", None).as_str() {
        "elemul" => Op::Elemul,
        "matmul" => Op::Matmul,
        "exp" => Op::Exp,
        "sigmoid" => Op::Sigmoid,
        "softmax" => Op::Softmax,
        "gelu" => Op::Gelu,
        _ => usage(),
    }
}

fn n_arg(args: &[String]) -> usize {
    arg_value(args, "--n", None).parse().expect("invalid --n")
}

fn k_arg(args: &[String]) -> usize {
    arg_value(args, "--k", Some("4")).parse().expect("invalid --k")
}

fn elemul_len(args: &[String]) -> usize {
    arg_value(args, "--len", None).parse().expect("invalid --len")
}

fn matmul_shape(args: &[String]) -> (usize, usize, usize) {
    let m = arg_value(args, "--m", None).parse().expect("invalid --m");
    let k = arg_value(args, "--k", None).parse().expect("invalid --k");
    let n = arg_value(args, "--n", None).parse().expect("invalid --n");
    (m, k, n)
}

fn timed_three<R0, R1, F0, F1, FH>(f0: F0, f1: F1, fh: FH) -> (f64, R0, R1)
where
    R0: Send + 'static,
    R1: Send + 'static,
    F0: FnOnce(PartyEndpoint) -> R0 + Send + 'static,
    F1: FnOnce(PartyEndpoint) -> R1 + Send + 'static,
    FH: FnOnce(HpEndpoint) + Send + 'static,
{
    let (p0, p1, hp) = make_mock();
    let barrier = Arc::new(Barrier::new(3));

    let b0 = Arc::clone(&barrier);
    let t0 = thread::spawn(move || {
        b0.wait();
        let start = Instant::now();
        let out = f0(p0);
        (start.elapsed().as_secs_f64(), out)
    });

    let b1 = Arc::clone(&barrier);
    let t1 = thread::spawn(move || {
        b1.wait();
        let start = Instant::now();
        let out = f1(p1);
        (start.elapsed().as_secs_f64(), out)
    });

    let bh = Arc::clone(&barrier);
    let th = thread::spawn(move || {
        bh.wait();
        let start = Instant::now();
        fh(hp);
        start.elapsed().as_secs_f64()
    });

    let (e0, o0) = t0.join().expect("p0 thread panicked");
    let (e1, o1) = t1.join().expect("p1 thread panicked");
    let eh = th.join().expect("hp thread panicked");
    (e0.max(e1).max(eh), o0, o1)
}

fn run_elemul(len: usize) -> (f64, String) {
    let (elapsed, out0, out1) = timed_three(
        move |comm| {
            let (x, y) = elemul_inputs(len);
            let x_share = share_vec(&x, 0xE005, 0);
            let y_share = share_vec(&y, 0xF006, 0);
            party_elemul(0, &x_share, &y_share, &mut PrgSync::new(&SHARED_SEED), &comm)
        },
        move |comm| {
            let (x, y) = elemul_inputs(len);
            let x_share = share_vec(&x, 0xE005, 1);
            let y_share = share_vec(&y, 0xF006, 1);
            party_elemul(1, &x_share, &y_share, &mut PrgSync::new(&SHARED_SEED), &comm)
        },
        move |comm| hp_elemul(len, &mut PrgSync::new(&HP_P0_SEED), &comm),
    );
    let (x, y) = elemul_inputs(len);
    for i in 0..len {
        assert_eq!(add(out0[i], out1[i]), mul(x[i], y[i]), "elemul mismatch at {i}");
    }
    (elapsed, format!("elemul ok: {len}/{len}"))
}

fn run_matmul(m: usize, k: usize, n: usize) -> (f64, String) {
    let (elapsed, out0, out1) = timed_three(
        move |comm| {
            let (a, b) = matmul_inputs(m, k, n);
            let a_share = share_vec(&a, 0xE105, 0);
            let b_share = share_vec(&b, 0xF106, 0);
            party_matmul(0, &a_share, &b_share, m, k, n, &mut PrgSync::new(&SHARED_SEED), &comm)
        },
        move |comm| {
            let (a, b) = matmul_inputs(m, k, n);
            let a_share = share_vec(&a, 0xE105, 1);
            let b_share = share_vec(&b, 0xF106, 1);
            party_matmul(1, &a_share, &b_share, m, k, n, &mut PrgSync::new(&SHARED_SEED), &comm)
        },
        move |comm| hp_matmul(m, k, n, &mut PrgSync::new(&HP_P0_SEED), &comm),
    );
    let (a, b) = matmul_inputs(m, k, n);
    for i in 0..m {
        for j in 0..n {
            let mut want = 0u64;
            for t in 0..k {
                want = add(want, mul(a[i * k + t], b[t * n + j]));
            }
            let idx = i * n + j;
            assert_eq!(add(out0[idx], out1[idx]), want, "matmul mismatch at ({i},{j})");
        }
    }
    (elapsed, format!("matmul ok: {}/{}", m * n, m * n))
}

fn party_nonlinear(op: Op, n: usize, k: usize, id: u8, comm: PartyEndpoint) -> NonlinearOut {
    let mut prg0 = PrgSync::new(&SHARED_SEED);
    match op {
        Op::Exp | Op::Sigmoid | Op::Gelu => {
            let shares: Vec<f64> = (0..n)
                .map(|i| share_value(value(i, op), i, id))
                .collect();
            let out = match op {
                Op::Exp => party_exp_batch(id, &shares, &mut prg0, &comm),
                Op::Sigmoid => party_sigmoid_batch(id, &shares, &mut prg0, &comm),
                Op::Gelu => party_gelu_batch(id, &shares, &mut prg0, &comm),
                _ => unreachable!(),
            };
            NonlinearOut::Reals(out)
        }
        Op::Softmax => {
            let shares: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let xs = vector_value(i, k);
                    xs.iter()
                        .enumerate()
                        .map(|(j, &v)| share_value(v, i * k + j, id))
                        .collect()
                })
                .collect();
            let out = party_softmax_batch(id, &shares, &mut prg0, &comm);
            NonlinearOut::Rows(out)
        }
        _ => unreachable!(),
    }
}

fn hp_nonlinear(op: Op, n: usize, k: usize, comm: HpEndpoint) {
    let mut asprg = PrgSync::new(&HP_P0_SEED);
    match op {
        Op::Exp => hp_exp_batch(n, &mut asprg, &comm),
        Op::Sigmoid => hp_sigmoid_batch(n, &mut asprg, &comm),
        Op::Softmax => hp_softmax_batch(n, k, &mut asprg, &comm),
        Op::Gelu => hp_gelu_batch(n, &mut asprg, &comm),
        _ => unreachable!(),
    }
}

fn run_nonlinear(op: Op, n: usize, k: usize) -> (f64, String) {
    let (elapsed, out0, out1) = timed_three(
        move |comm| party_nonlinear(op, n, k, 0, comm),
        move |comm| party_nonlinear(op, n, k, 1, comm),
        move |comm| hp_nonlinear(op, n, k, comm),
    );
    let mut max_err = 0.0f64;
    match (out0, out1) {
        (NonlinearOut::Reals(v0), NonlinearOut::Reals(v1)) => {
            for i in 0..n {
                let x = value(i, op);
                let want = match op {
                    Op::Exp => x.exp(),
                    Op::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                    Op::Gelu => x * (1.0 / (1.0 + (-(GELU_COEF * x)).exp())),
                    _ => unreachable!(),
                };
                max_err = max_err.max((v0[i] + v1[i] - want).abs());
            }
        }
        (NonlinearOut::Rows(r0), NonlinearOut::Rows(r1)) => {
            for i in 0..n {
                let xs = vector_value(i, k);
                let exps: Vec<f64> = xs.iter().map(|v| v.exp()).collect();
                let denom: f64 = exps.iter().sum();
                for j in 0..k {
                    max_err = max_err.max((r0[i][j] + r1[i][j] - exps[j] / denom).abs());
                }
            }
        }
        _ => panic!("nonlinear output type mismatch"),
    }
    (elapsed, format!("{op:?} ok: n={n}, max_err={max_err:.3e}"))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let op = parse_op(&args[1..]);
    let (elapsed, verify) = match op {
        Op::Elemul => run_elemul(elemul_len(&args[1..])),
        Op::Matmul => {
            let (m, k, n) = matmul_shape(&args[1..]);
            run_matmul(m, k, n)
        }
        Op::Exp | Op::Sigmoid | Op::Gelu => run_nonlinear(op, n_arg(&args[1..]), 0),
        Op::Softmax => run_nonlinear(op, n_arg(&args[1..]), k_arg(&args[1..])),
    };
    println!("[mcu-thread] done: {elapsed:.9}s");
    println!("[mcu-thread] verify: {verify}");
}
