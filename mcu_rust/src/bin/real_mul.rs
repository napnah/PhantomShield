use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use mcu_rust::channel::{SocketHpEndpoint, SocketPartyEndpoint};
use mcu_rust::prg::PrgSync;
use mcu_rust::protocols::multiply::{hp_multiply, party_multiply};
use mcu_rust::ring::{add, sub};

const SHARED_SEED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const HP_P0_SEED: [u8; 16] = [
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

#[derive(Clone, Copy)]
struct Case {
    x: u64,
    y: u64,
    x0: u64,
    y0: u64,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn make_case(i: usize) -> Case {
    let mut s = 0x1234_5678_9ABC_DEF0u64 ^ i as u64;
    Case {
        x: splitmix64(&mut s),
        y: splitmix64(&mut s),
        x0: splitmix64(&mut s),
        y0: splitmix64(&mut s),
    }
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  real_mul hp --addr 127.0.0.1:9100 --n 10000\n  real_mul p0 --addr 127.0.0.1:9100 --n 10000 --out p0.shares\n  real_mul p1 --addr 127.0.0.1:9100 --n 10000 --out p1.shares\n  real_mul verify --n 10000 --p0 p0.shares --p1 p1.shares"
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

fn run_hp(args: &[String]) -> std::io::Result<()> {
    let addr = arg_value(args, "--addr", Some("127.0.0.1:9100"));
    let n: usize = arg_value(args, "--n", None).parse().expect("invalid --n");
    println!("[HP] listening on {addr}, n={n}");
    let comm = SocketHpEndpoint::listen(&addr)?;
    println!("[HP] p0 and p1 connected");
    let mut asprg = PrgSync::new(&HP_P0_SEED);
    let start = Instant::now();
    for _ in 0..n {
        hp_multiply(&mut asprg, &comm);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "[HP] done: {:.6}s, {:.0} mul/s",
        elapsed,
        n as f64 / elapsed
    );
    Ok(())
}

fn run_party(args: &[String], id: u8) -> std::io::Result<()> {
    let addr = arg_value(args, "--addr", Some("127.0.0.1:9100"));
    let n: usize = arg_value(args, "--n", None).parse().expect("invalid --n");
    let out = arg_value(
        args,
        "--out",
        Some(if id == 0 { "p0.shares" } else { "p1.shares" }),
    );
    let comm = SocketPartyEndpoint::connect(&addr, id)?;
    let mut prg = PrgSync::new(&SHARED_SEED);
    let mut writer = BufWriter::new(File::create(&out)?);
    let start = Instant::now();

    for i in 0..n {
        let c = make_case(i);
        let x_share = if id == 0 { c.x0 } else { sub(c.x, c.x0) };
        let y_share = if id == 0 { c.y0 } else { sub(c.y, c.y0) };
        let z_share = party_multiply(id, x_share, y_share, &mut prg, &comm);
        writeln!(writer, "{z_share}")?;
    }

    writer.flush()?;
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "[P{id}] done: {:.6}s, {:.0} mul/s, wrote {}",
        elapsed,
        n as f64 / elapsed,
        out
    );
    Ok(())
}

fn read_shares(path: &str) -> std::io::Result<Vec<u64>> {
    let file = File::open(path)?;
    let mut shares = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        shares.push(line.trim().parse().expect("invalid share line"));
    }
    Ok(shares)
}

fn run_verify(args: &[String]) -> std::io::Result<()> {
    let n: usize = arg_value(args, "--n", None).parse().expect("invalid --n");
    let p0_path = arg_value(args, "--p0", Some("p0.shares"));
    let p1_path = arg_value(args, "--p1", Some("p1.shares"));
    let p0 = read_shares(&p0_path)?;
    let p1 = read_shares(&p1_path)?;
    if p0.len() != n || p1.len() != n {
        eprintln!(
            "[verify] length mismatch: expected {n}, got p0={}, p1={}",
            p0.len(),
            p1.len()
        );
        std::process::exit(1);
    }

    let mut bad = 0usize;
    for i in 0..n {
        let c = make_case(i);
        let got = add(p0[i], p1[i]);
        let want = c.x.wrapping_mul(c.y);
        if got != want {
            bad += 1;
            if bad <= 5 {
                eprintln!("[verify] mismatch at {i}: got={got}, want={want}");
            }
        }
    }
    if bad > 0 {
        eprintln!("[verify] failed: {bad}/{n} mismatches");
        std::process::exit(1);
    }
    println!("[verify] ok: {n}/{n} products matched");
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
