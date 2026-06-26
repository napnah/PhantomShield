use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use mcu_rust::channel::{
    reset_socket_stats, socket_stats_snapshot, SocketHpEndpoint, SocketPartyEndpoint,
};
use mcu_rust::prg::PrgSync;
use mcu_rust::protocols::GELU_COEF;
use mcu_rust::real_protocols::{
    hp_exp, hp_gelu, hp_sigmoid, hp_sign_bicoptor, hp_sign_ge_zero, hp_softmax, hp_wrap,
    hp_wrap_bicoptor, party_exp, party_gelu, party_sigmoid, party_sign_bicoptor,
    party_sign_ge_zero, party_softmax_all, party_wrap, party_wrap_bicoptor,
};

const SHARED_SEED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const HP_P0_SEED: [u8; 16] = [
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Sign,
    SignBicoptor,
    Wrap,
    WrapBicoptor,
    Exp,
    Sigmoid,
    Softmax,
    Gelu,
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

fn value(i: usize, op: Op) -> f64 {
    let mut s = 0xCAFE_BABE_D15E_A5E5u64 ^ ((i as u64) << 7);
    match op {
        Op::Wrap | Op::WrapBicoptor => -200.0 + unit(&mut s) * 400.0,
        Op::Sign => -200.0 + unit(&mut s) * 400.0,
        Op::SignBicoptor => unreachable!("sign-bicoptor uses integer_value"),
        Op::Exp => -8.0 + unit(&mut s) * 16.0,
        Op::Sigmoid => -8.0 + unit(&mut s) * 16.0,
        Op::Gelu => -5.0 + unit(&mut s) * 10.0,
        Op::Softmax => unreachable!("softmax uses vector_value"),
    }
}

fn vector_value(i: usize, k: usize) -> Vec<f64> {
    let mut s = 0x534F_F74D_AA55_1001u64 ^ ((i as u64) << 9);
    (0..k).map(|_| -8.0 + unit(&mut s) * 16.0).collect()
}

fn integer_value(i: usize, lx: u32) -> i64 {
    let limit = 1i64 << (lx - 1);
    let mut s = 0x5151_900D_AAAA_0001u64 ^ ((i as u64) << 11);
    let mag = (splitmix64(&mut s) % (limit as u64 - 1) + 1) as i64;
    if splitmix64(&mut s) & 1 == 0 { mag } else { -mag }
}

fn share_int(v: i64, i: usize, id: u8) -> u64 {
    let mut s = 0x7777_2222_1111_0000u64 ^ ((i as u64) << 3);
    let s0 = splitmix64(&mut s);
    let vu = v as u64;
    if id == 0 { s0 } else { vu.wrapping_sub(s0) }
}

fn share_value(v: f64, i: usize, id: u8) -> f64 {
    let mut s = 0x5123_4567_89AB_CDEFu64 ^ ((i as u64) << 5);
    let s0 = -50.0 + unit(&mut s) * 100.0;
    if id == 0 {
        s0
    } else {
        v - s0
    }
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  real_nonlinear hp --op exp --addr 127.0.0.1:9300 --n 100\n  real_nonlinear p0 --op exp --addr 127.0.0.1:9300 --n 100 --out p0.out\n  real_nonlinear p1 --op exp --addr 127.0.0.1:9300 --n 100 --out p1.out\n  real_nonlinear verify --op exp --n 100 --p0 p0.out --p1 p1.out\n\n  ops: sign, sign-bicoptor, wrap, wrap-bicoptor, exp, sigmoid, softmax, gelu. For softmax add --k 8; for sign-bicoptor add --lx 16"
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
        "wrap" => Op::Wrap,
        "wrap-bicoptor" => Op::WrapBicoptor,
        "sign" => Op::Sign,
        "sign-bicoptor" => Op::SignBicoptor,
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
    arg_value(args, "--k", Some("8"))
        .parse()
        .expect("invalid --k")
}

fn lx_arg(args: &[String]) -> u32 {
    arg_value(args, "--lx", Some("16"))
        .parse()
        .expect("invalid --lx")
}

fn run_hp(args: &[String]) -> std::io::Result<()> {
    let op = parse_op(args);
    let n = n_arg(args);
    let k = k_arg(args);
    let lx = lx_arg(args);
    let addr = arg_value(args, "--addr", Some("127.0.0.1:9300"));
    println!("[HP] listening on {addr}, op={op:?}, n={n}");
    let comm = SocketHpEndpoint::listen(&addr)?;
    println!("[HP] p0 and p1 connected");
    let mut asprg = PrgSync::new(&HP_P0_SEED);
    reset_socket_stats();
    let start = Instant::now();
    for _ in 0..n {
        match op {
            Op::Wrap => hp_wrap(&comm),
            Op::WrapBicoptor => hp_wrap_bicoptor(&comm),
            Op::Sign => hp_sign_ge_zero(&comm),
            Op::SignBicoptor => hp_sign_bicoptor(lx, &comm),
            Op::Exp => hp_exp(&mut asprg, &comm),
            Op::Sigmoid => hp_sigmoid(&mut asprg, &comm),
            Op::Softmax => hp_softmax(k, &mut asprg, &comm),
            Op::Gelu => hp_gelu(&mut asprg, &comm),
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("[HP] done: {:.6}s, {:.0} op/s", elapsed, n as f64 / elapsed);
    print_timing("hp", elapsed);
    Ok(())
}

fn run_party(args: &[String], id: u8) -> std::io::Result<()> {
    let op = parse_op(args);
    let n = n_arg(args);
    let k = k_arg(args);
    let addr = arg_value(args, "--addr", Some("127.0.0.1:9300"));
    let out = arg_value(
        args,
        "--out",
        Some(if id == 0 { "p0.nonlinear" } else { "p1.nonlinear" }),
    );
    let comm = SocketPartyEndpoint::connect(&addr, id)?;
    let mut prg0 = PrgSync::new(&SHARED_SEED);
    let mut writer = BufWriter::new(File::create(&out)?);
    reset_socket_stats();
    let start = Instant::now();
    for i in 0..n {
        match op {
            Op::Wrap => {
                let x = value(i, op);
                let share = share_value(x, i, id);
                let r = 7.0 + ((i * 31) % 240) as f64;
                let w = party_wrap(id, share, r, 256.0, &mut prg0, &comm);
                writeln!(writer, "{w}")?;
            }
            Op::WrapBicoptor => {
                let x = value(i, op);
                let share = share_value(x, i, id);
                let r = 7.0 + ((i * 31) % 240) as f64;
                let w = party_wrap_bicoptor(id, share, r, 256.0, &mut prg0, &comm);
                writeln!(writer, "{w}")?;
            }
            Op::Sign => {
                let x = value(i, op);
                let share = share_value(x, i, id);
                let bit = party_sign_ge_zero(share, &mut prg0, &comm);
                writeln!(writer, "{bit}")?;
            }
            Op::SignBicoptor => {
                let lx = lx_arg(args);
                let x = integer_value(i, lx);
                let share = share_int(x, i, id);
                let bit = party_sign_bicoptor(id, share, lx, &mut prg0, &comm);
                writeln!(writer, "{bit}")?;
            }
            Op::Exp => {
                let x = value(i, op);
                let share = share_value(x, i, id);
                let y = party_exp(id, share, &mut prg0, &comm);
                writeln!(writer, "{y:.17e}")?;
            }
            Op::Sigmoid => {
                let x = value(i, op);
                let share = share_value(x, i, id);
                let y = party_sigmoid(id, share, &mut prg0, &comm);
                writeln!(writer, "{y:.17e}")?;
            }
            Op::Softmax => {
                let xs = vector_value(i, k);
                let shares: Vec<f64> = xs
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| share_value(v, i * k + j, id))
                    .collect();
                let vals: Vec<String> = party_softmax_all(id, &shares, &mut prg0, &comm)
                    .into_iter()
                    .map(|y| format!("{y:.17e}"))
                    .collect();
                writeln!(writer, "{}", vals.join(","))?;
            }
            Op::Gelu => {
                let x = value(i, op);
                let share = share_value(x, i, id);
                let y = party_gelu(id, share, &mut prg0, &comm);
                writeln!(writer, "{y:.17e}")?;
            }
        }
    }
    let protocol_elapsed = start.elapsed().as_secs_f64();
    let write_start = Instant::now();
    writer.flush()?;
    let write_elapsed = write_start.elapsed().as_secs_f64();
    let elapsed = protocol_elapsed + write_elapsed;
    println!("[P{id}] done: {:.6}s, {:.0} op/s, wrote {out}", elapsed, n as f64 / elapsed);
    print_timing(&format!("p{id}"), elapsed);
    println!(
        "[P{id}] timing_breakdown protocol_s={protocol_elapsed:.9} write_s={write_elapsed:.9}"
    );
    Ok(())
}

fn print_timing(role: &str, total_s: f64) {
    let stats = socket_stats_snapshot();
    let send_s = stats.send_nanos as f64 / 1e9;
    let recv_s = stats.recv_nanos as f64 / 1e9;
    let comm_s = send_s + recv_s;
    let local_s = (total_s - comm_s).max(0.0);
    println!(
        "[{role}] timing total_s={total_s:.9} comm_s={comm_s:.9} local_s={local_s:.9} send_s={send_s:.9} recv_s={recv_s:.9} send_msgs={} recv_msgs={} send_bytes={} recv_bytes={}",
        stats.send_messages,
        stats.recv_messages,
        stats.send_bytes,
        stats.recv_bytes
    );
}

fn read_lines(path: &str) -> std::io::Result<Vec<String>> {
    let file = File::open(path)?;
    BufReader::new(file).lines().collect()
}

fn verify(args: &[String]) -> std::io::Result<()> {
    let op = parse_op(args);
    let n = n_arg(args);
    let k = k_arg(args);
    let lx = lx_arg(args);
    let p0 = read_lines(&arg_value(args, "--p0", Some("p0.nonlinear")))?;
    let p1 = read_lines(&arg_value(args, "--p1", Some("p1.nonlinear")))?;
    assert_eq!(p0.len(), n, "p0 length mismatch");
    assert_eq!(p1.len(), n, "p1 length mismatch");
    let mut max_err = 0.0f64;
    for i in 0..n {
        match op {
            Op::Sign => {
                let got: u8 = p0[i].parse::<u8>().unwrap();
                let got1: u8 = p1[i].parse::<u8>().unwrap();
                let x = value(i, op);
                let want = if x >= 0.0 { 1 } else { 0 };
                if got != want || got1 != want {
                    eprintln!("sign mismatch at {i}: got=({got},{got1}) want={want}");
                    std::process::exit(1);
                }
            }
            Op::SignBicoptor => {
                let got: u8 = p0[i].parse::<u8>().unwrap();
                let got1: u8 = p1[i].parse::<u8>().unwrap();
                let x = integer_value(i, lx);
                let want = if x >= 0 { 1 } else { 0 };
                if got != want || got1 != want {
                    eprintln!("sign-bicoptor mismatch at {i}: got=({got},{got1}) want={want} x={x}");
                    std::process::exit(1);
                }
            }
            Op::Wrap | Op::WrapBicoptor => {
                let got: i64 = p0[i].parse::<i64>().unwrap();
                let got1: i64 = p1[i].parse::<i64>().unwrap();
                let x = value(i, op);
                let r = 7.0 + ((i * 31) % 240) as f64;
                let want = ((x + r) / 256.0).floor() as i64;
                if got != want || got1 != want {
                    eprintln!("wrap mismatch at {i}: got=({got},{got1}) want={want}");
                    std::process::exit(1);
                }
            }
            Op::Exp | Op::Sigmoid | Op::Gelu => {
                let y0: f64 = p0[i].parse().unwrap();
                let y1: f64 = p1[i].parse().unwrap();
                let x = value(i, op);
                let want = match op {
                    Op::Exp => x.exp(),
                    Op::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                    Op::Gelu => x * (1.0 / (1.0 + (-(GELU_COEF * x)).exp())),
                    _ => unreachable!(),
                };
                max_err = max_err.max((y0 + y1 - want).abs());
            }
            Op::Softmax => {
                let y0: Vec<f64> = p0[i].split(',').map(|v| v.parse().unwrap()).collect();
                let y1: Vec<f64> = p1[i].split(',').map(|v| v.parse().unwrap()).collect();
                let xs = vector_value(i, k);
                let exps: Vec<f64> = xs.iter().map(|v| v.exp()).collect();
                let denom: f64 = exps.iter().sum();
                for j in 0..k {
                    max_err = max_err.max((y0[j] + y1[j] - exps[j] / denom).abs());
                }
            }
        }
    }
    println!("[verify] {op:?} ok: n={n}, max_err={max_err:.3e}");
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
        "verify" => verify(&args[2..]),
        _ => usage(),
    }
}
