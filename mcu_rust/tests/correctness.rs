//! 集成测试：对齐 Python `verify_*` 的容差指标，并验证 Comm 路径与融合路径一致。

use std::thread;

use mcu_rust::channel::make_mock;
use mcu_rust::prg::PrgSync;
use mcu_rust::protocols::multiply::{hp_multiply, party_multiply};
use mcu_rust::ring::add;
use mcu_rust::simulate;

fn seed_lo() -> [u8; 16] {
    let mut s = [0u8; 16];
    for (i, b) in s.iter_mut().enumerate() {
        *b = i as u8;
    }
    s
}
fn seed_hi() -> [u8; 16] {
    let mut s = [0u8; 16];
    for (i, b) in s.iter_mut().enumerate() {
        *b = (i + 16) as u8;
    }
    s
}

// 简单可复现的伪随机（仅测试造数用，与协议 PRG 无关）
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

#[test]
fn multiply_batch_exact() {
    let n = 1000;
    let mut st = 1u64;
    let (mut x0, mut x1, mut y0, mut y1) = (vec![], vec![], vec![], vec![]);
    let mut expected = vec![];
    for _ in 0..n {
        let x = lcg(&mut st) % 1_000_000_000;
        let y = lcg(&mut st) % 1_000_000_000;
        let a = lcg(&mut st);
        let b = lcg(&mut st);
        x0.push(a);
        x1.push(x.wrapping_sub(a));
        y0.push(b);
        y1.push(y.wrapping_sub(b));
        expected.push(x.wrapping_mul(y));
    }
    let (s0, s1) = simulate::multiply_batch(&x0, &x1, &y0, &y1, &seed_lo(), &seed_hi());
    for i in 0..n {
        assert_eq!(add(s0[i], s1[i]), expected[i], "乘法第 {i} 个不精确");
    }
}

#[test]
fn multiply_comm_path_matches() {
    // 通过 Comm + 三线程跑一次乘法，验证抽象层正确
    let x: u64 = 12345;
    let y: u64 = 67890;
    let x0: u64 = 999_999;
    let x1 = x.wrapping_sub(x0);
    let y0: u64 = 888_888;
    let y1 = y.wrapping_sub(y0);

    let (cp0, cp1, chp) = make_mock();
    let mut prg0_p0 = PrgSync::new(&seed_lo());
    let mut prg0_p1 = PrgSync::new(&seed_lo());
    let mut asprg = PrgSync::new(&seed_hi());

    let h0 = thread::spawn(move || party_multiply(0, x0, y0, &mut prg0_p0, &cp0));
    let h1 = thread::spawn(move || party_multiply(1, x1, y1, &mut prg0_p1, &cp1));
    let hh = thread::spawn(move || hp_multiply(&mut asprg, &chp));

    let r0 = h0.join().unwrap();
    let r1 = h1.join().unwrap();
    hh.join().unwrap();

    assert_eq!(add(r0, r1), x.wrapping_mul(y));
}

#[test]
fn exp_avg_err_below_1e4() {
    let n = 5000;
    let mut st = 7u64;
    let (mut x0, mut x1) = (vec![], vec![]);
    let mut xs = vec![];
    for _ in 0..n {
        let x = (lcg(&mut st) as f64 / u64::MAX as f64) * 20.0 - 10.0; // U[-10,10]
        let sp = (lcg(&mut st) as f64 / u64::MAX as f64) * 200.0 - 100.0;
        x0.push(sp);
        x1.push(x - sp);
        xs.push(x);
    }
    let (e0, e1) = simulate::exp_batch(&x0, &x1, &seed_lo(), &seed_hi());
    let mut sum_abs = 0.0;
    for i in 0..n {
        let want = xs[i].exp();
        sum_abs += (e0[i] + e1[i] - want).abs();
    }
    let avg = sum_abs / n as f64;
    assert!(avg < 1e-4, "exp 平均绝对误差 {avg} 未达标");
}

#[test]
fn sigmoid_max_err_below_1e4() {
    let n = 5000;
    let mut st = 11u64;
    let (mut z0, mut z1) = (vec![], vec![]);
    let mut zs = vec![];
    for _ in 0..n {
        let z = (lcg(&mut st) as f64 / u64::MAX as f64) * 16.0 - 8.0; // U[-8,8]
        let sp = (lcg(&mut st) as f64 / u64::MAX as f64) * 100.0 - 50.0;
        z0.push(sp);
        z1.push(z - sp);
        zs.push(z);
    }
    let (s0, s1) = simulate::sigmoid_batch(&z0, &z1, &seed_lo(), &seed_hi());
    let mut max_abs: f64 = 0.0;
    for i in 0..n {
        let want = 1.0 / (1.0 + (-zs[i]).exp());
        max_abs = max_abs.max((s0[i] + s1[i] - want).abs());
    }
    assert!(max_abs < 1e-4, "sigmoid 最大绝对误差 {max_abs} 未达标");
}

#[test]
fn gelu_avg_err_below_1e3() {
    let n = 5000;
    let mut st = 13u64;
    let (mut x0, mut x1) = (vec![], vec![]);
    let mut xs = vec![];
    for _ in 0..n {
        let x = (lcg(&mut st) as f64 / u64::MAX as f64) * 10.0 - 5.0; // U[-5,5]
        let sp = (lcg(&mut st) as f64 / u64::MAX as f64) * 100.0 - 50.0;
        x0.push(sp);
        x1.push(x - sp);
        xs.push(x);
    }
    let (g0, g1) = simulate::gelu_batch(&x0, &x1, &seed_lo(), &seed_hi());
    let mut sum_abs = 0.0;
    for i in 0..n {
        let s = 1.0 / (1.0 + (-(1.702 * xs[i])).exp());
        let want = xs[i] * s;
        sum_abs += (g0[i] + g1[i] - want).abs();
    }
    let avg = sum_abs / n as f64;
    assert!(avg < 1e-3, "gelu 平均绝对误差 {avg} 未达标");
}

#[test]
fn softmax_max_err_below_1e4() {
    let n = 200;
    let k = 8;
    let mut st = 17u64;
    let mut x0 = vec![0.0; n * k];
    let mut x1 = vec![0.0; n * k];
    let mut xs = vec![0.0; n * k];
    for i in 0..n {
        for j in 0..k {
            let v = (lcg(&mut st) as f64 / u64::MAX as f64) * 16.0 - 8.0; // U[-8,8]
            let sp = (lcg(&mut st) as f64 / u64::MAX as f64) * 100.0 - 50.0;
            xs[i * k + j] = v;
            x0[i * k + j] = sp;
            x1[i * k + j] = v - sp;
        }
    }
    let (s0, s1) = simulate::softmax_batch(&x0, &x1, n, k, &seed_lo(), &seed_hi());
    let mut max_abs: f64 = 0.0;
    for i in 0..n {
        let row = &xs[i * k..i * k + k];
        let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|v| (v - mx).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for j in 0..k {
            let want = exps[j] / sum;
            max_abs = max_abs.max((s0[i * k + j] + s1[i * k + j] - want).abs());
        }
    }
    assert!(max_abs < 1e-4, "softmax 最大绝对误差 {max_abs} 未达标");
}
