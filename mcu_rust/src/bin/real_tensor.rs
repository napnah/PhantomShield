use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use mcu_rust::channel::{
    reset_socket_stats, socket_stats_snapshot, SocketHpEndpoint, SocketPartyEndpoint,
};
use mcu_rust::prg::PrgSync;
use mcu_rust::ring::{add, mul, sub};
use mcu_rust::tensor::{
    hp_elemul, hp_matmul, party_elemul, party_matmul, reset_tensor_stats, tensor_stats_snapshot,
};

const SHARED_SEED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const HP_P0_SEED: [u8; 16] = [
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Elemul,
    Matmul,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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

fn usage() -> ! {
    eprintln!(
        "usage:\n  real_tensor hp --op elemul --addr 127.0.0.1:9200 --len 10000\n  real_tensor p0 --op elemul --addr 127.0.0.1:9200 --len 10000 --out p0.out\n  real_tensor p1 --op elemul --addr 127.0.0.1:9200 --len 10000 --out p1.out\n  real_tensor verify --op elemul --len 10000 --p0 p0.out --p1 p1.out\n\n  real_tensor hp --op matmul --addr 127.0.0.1:9200 --m 16 --k 64 --n 16\n  real_tensor p0 --op matmul --addr 127.0.0.1:9200 --m 16 --k 64 --n 16 --out p0.out\n  real_tensor p1 --op matmul --addr 127.0.0.1:9200 --m 16 --k 64 --n 16 --out p1.out\n  real_tensor verify --op matmul --m 16 --k 64 --n 16 --p0 p0.out --p1 p1.out"
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
        _ => usage(),
    }
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

fn write_vec(path: &str, values: &[u64]) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for v in values {
        writeln!(writer, "{v}")?;
    }
    writer.flush()
}

fn read_vec(path: &str) -> std::io::Result<Vec<u64>> {
    let file = File::open(path)?;
    let mut values = Vec::new();
    for line in BufReader::new(file).lines() {
        values.push(line?.trim().parse().expect("invalid share line"));
    }
    Ok(values)
}

fn elemul_inputs(len: usize) -> (Vec<u64>, Vec<u64>) {
    (make_vec(0xA001, len), make_vec(0xB002, len))
}

fn matmul_inputs(m: usize, k: usize, n: usize) -> (Vec<u64>, Vec<u64>) {
    (make_vec(0xC003, m * k), make_vec(0xD004, k * n))
}

fn run_hp(args: &[String]) -> std::io::Result<()> {
    let op = parse_op(args);
    let addr = arg_value(args, "--addr", Some("127.0.0.1:9200"));
    println!("[HP] listening on {addr}, op={op:?}");
    let comm = SocketHpEndpoint::listen(&addr)?;
    println!("[HP] p0 and p1 connected");
    let mut asprg = PrgSync::new(&HP_P0_SEED);
    reset_socket_stats();
    reset_tensor_stats();
    let start = Instant::now();
    let mul_count = match op {
        Op::Elemul => {
            let len = elemul_len(args);
            hp_elemul(len, &mut asprg, &comm);
            len
        }
        Op::Matmul => {
            let (m, k, n) = matmul_shape(args);
            hp_matmul(m, k, n, &mut asprg, &comm);
            m * k * n
        }
    };
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "[HP] done: {:.6}s, {:.0} secure mul/s",
        elapsed,
        mul_count as f64 / elapsed
    );
    print_timing("hp", elapsed);
    Ok(())
}

fn run_party(args: &[String], id: u8) -> std::io::Result<()> {
    let op = parse_op(args);
    let addr = arg_value(args, "--addr", Some("127.0.0.1:9200"));
    let out_path = arg_value(
        args,
        "--out",
        Some(if id == 0 { "p0.tensor" } else { "p1.tensor" }),
    );
    let comm = SocketPartyEndpoint::connect(&addr, id)?;
    let mut prg = PrgSync::new(&SHARED_SEED);
    reset_socket_stats();
    reset_tensor_stats();
    let setup_start = Instant::now();
    let (out, mul_count) = match op {
        Op::Elemul => {
            let len = elemul_len(args);
            let (x, y) = elemul_inputs(len);
            let x_share = share_vec(&x, 0xE005, id);
            let y_share = share_vec(&y, 0xF006, id);
            (party_elemul(id, &x_share, &y_share, &mut prg, &comm), len)
        }
        Op::Matmul => {
            let (m, k, n) = matmul_shape(args);
            let (a, b) = matmul_inputs(m, k, n);
            let a_share = share_vec(&a, 0xE105, id);
            let b_share = share_vec(&b, 0xF106, id);
            (
                party_matmul(id, &a_share, &b_share, m, k, n, &mut prg, &comm),
                m * k * n,
            )
        }
    };
    let protocol_elapsed = setup_start.elapsed().as_secs_f64();
    let write_start = Instant::now();
    write_vec(&out_path, &out)?;
    let write_elapsed = write_start.elapsed().as_secs_f64();
    println!(
        "[P{id}] done: {:.6}s, {:.0} secure mul/s, wrote {} values to {}",
        protocol_elapsed,
        mul_count as f64 / protocol_elapsed,
        out.len(),
        out_path
    );
    print_timing(&format!("p{id}"), protocol_elapsed);
    println!(
        "[P{id}] timing_breakdown protocol_s={protocol_elapsed:.9} write_s={write_elapsed:.9}"
    );
    Ok(())
}

fn print_timing(role: &str, total_s: f64) {
    let stats = socket_stats_snapshot();
    let tensor_stats = tensor_stats_snapshot();
    let send_s = stats.send_nanos as f64 / 1e9;
    let recv_s = stats.recv_nanos as f64 / 1e9;
    let recv_wait_s = stats.recv_wait_nanos as f64 / 1e9;
    let recv_read_s = stats.recv_read_nanos as f64 / 1e9;
    let comm_s = send_s + recv_s;
    let compute_s = tensor_stats.local_nanos as f64 / 1e9;
    let other_local_s = (total_s - comm_s - compute_s).max(0.0);
    let local_s = (total_s - comm_s).max(0.0);
    println!(
        "[{role}] timing total_s={total_s:.9} comm_s={comm_s:.9} local_s={local_s:.9} compute_s={compute_s:.9} other_local_s={other_local_s:.9} send_s={send_s:.9} recv_s={recv_s:.9} recv_wait_s={recv_wait_s:.9} recv_read_s={recv_read_s:.9} send_msgs={} recv_msgs={} send_bytes={} recv_bytes={} matmul_calls={} cpu_matmul_calls={} cuda_matmul_calls={} fused_party_calls={} fused_hp_calls={} cuda_fallbacks={}",
        stats.send_messages,
        stats.recv_messages,
        stats.send_bytes,
        stats.recv_bytes,
        tensor_stats.matmul_calls,
        tensor_stats.cpu_matmul_calls,
        tensor_stats.cuda_matmul_calls,
        tensor_stats.fused_party_calls,
        tensor_stats.fused_hp_calls,
        tensor_stats.cuda_fallbacks,
    );
}

fn verify_elemul(args: &[String], p0: &[u64], p1: &[u64]) {
    let len = elemul_len(args);
    assert_eq!(p0.len(), len, "p0 length mismatch");
    assert_eq!(p1.len(), len, "p1 length mismatch");
    let (x, y) = elemul_inputs(len);
    let mut bad = 0usize;
    for i in 0..len {
        let got = add(p0[i], p1[i]);
        let want = mul(x[i], y[i]);
        if got != want {
            bad += 1;
            if bad <= 5 {
                eprintln!("[verify] elemul mismatch at {i}: got={got}, want={want}");
            }
        }
    }
    if bad > 0 {
        eprintln!("[verify] failed: {bad}/{len} mismatches");
        std::process::exit(1);
    }
    println!("[verify] elemul ok: {len}/{len}");
}

fn verify_matmul(args: &[String], p0: &[u64], p1: &[u64]) {
    let (m, k, n) = matmul_shape(args);
    assert_eq!(p0.len(), m * n, "p0 length mismatch");
    assert_eq!(p1.len(), m * n, "p1 length mismatch");
    let (a, b) = matmul_inputs(m, k, n);
    let mut bad = 0usize;
    for i in 0..m {
        for j in 0..n {
            let mut want = 0u64;
            for t in 0..k {
                want = add(want, mul(a[i * k + t], b[t * n + j]));
            }
            let idx = i * n + j;
            let got = add(p0[idx], p1[idx]);
            if got != want {
                bad += 1;
                if bad <= 5 {
                    eprintln!("[verify] matmul mismatch at ({i},{j}): got={got}, want={want}");
                }
            }
        }
    }
    if bad > 0 {
        eprintln!("[verify] failed: {bad}/{} mismatches", m * n);
        std::process::exit(1);
    }
    println!("[verify] matmul ok: {}/{}", m * n, m * n);
}

fn run_verify(args: &[String]) -> std::io::Result<()> {
    let op = parse_op(args);
    let p0_path = arg_value(args, "--p0", Some("p0.tensor"));
    let p1_path = arg_value(args, "--p1", Some("p1.tensor"));
    let p0 = read_vec(&p0_path)?;
    let p1 = read_vec(&p1_path)?;
    match op {
        Op::Elemul => verify_elemul(args, &p0, &p1),
        Op::Matmul => verify_matmul(args, &p0, &p1),
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
    }
    match args[1].as_str() {
        "hp" => run_hp(&args[2..]),
        "p0" => run_party(&args[2..], 0),
        "p1" => run_party(&args[2..], 1),
        "verify" => run_verify(&args[2..]),
        _ => usage(),
    }
}
