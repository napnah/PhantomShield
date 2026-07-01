use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use mcu_rust::channel::{
    reset_socket_stats, socket_stats_snapshot, HpComm, PartyComm, SocketHpEndpoint,
    SocketPartyEndpoint,
};
use mcu_rust::prg::PrgSync;
use mcu_rust::protocols::MOD;
use mcu_rust::real_protocols::{
    hp_gelu_batch, hp_softmax_batch, party_gelu_batch, party_softmax_batch,
};
use mcu_rust::ring::{add, sub};
use mcu_rust::tensor::{hp_matmul, party_matmul, reset_tensor_stats, tensor_stats_snapshot};

const SHARED_SEED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const HP_P0_SEED: [u8; 16] = [
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];
static MODEL_SHARE_CACHE: OnceLock<Mutex<HashMap<PathBuf, Vec<u64>>>> = OnceLock::new();

#[derive(Clone, Copy)]
struct BertShape {
    batch: usize,
    seq: usize,
    hidden: usize,
    heads: usize,
    ffn: usize,
    layers: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StateMode {
    Synthetic,
    Chained,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Synthetic,
    RealIo,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RescaleMode {
    Local,
    HpClear,
}

impl RescaleMode {
    fn as_str(self) -> &'static str {
        match self {
            RescaleMode::Local => "local",
            RescaleMode::HpClear => "hp_clear",
        }
    }
}

impl InputMode {
    fn as_str(self) -> &'static str {
        match self {
            InputMode::Synthetic => "synthetic",
            InputMode::RealIo => "real_io",
        }
    }
}

impl StateMode {
    fn as_str(self) -> &'static str {
        match self {
            StateMode::Synthetic => "synthetic",
            StateMode::Chained => "chained",
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: bert_session hp|p0|p1|service-hp|service-p0|service-p1 --addr HOST:PORT --batch 1 --seq 16 --hidden 768 --heads 12 --ffn 3072 --layers 12 [--state-mode synthetic|chained] [--input-mode synthetic|real_io] [--share-dir DIR] [--model-share-dir DIR] [--out p0.out] [--service-dir DIR]"
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

fn write_text_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, text)?;
    fs::rename(tmp, path)
}

fn parse_shape(args: &[String]) -> BertShape {
    let batch = arg_value(args, "--batch", Some("1"))
        .parse()
        .expect("invalid --batch");
    let seq = arg_value(args, "--seq", Some("16"))
        .parse()
        .expect("invalid --seq");
    let hidden = arg_value(args, "--hidden", Some("768"))
        .parse()
        .expect("invalid --hidden");
    let heads = arg_value(args, "--heads", Some("12"))
        .parse()
        .expect("invalid --heads");
    let ffn = arg_value(args, "--ffn", Some("3072"))
        .parse()
        .expect("invalid --ffn");
    let layers = arg_value(args, "--layers", Some("12"))
        .parse()
        .expect("invalid --layers");
    assert!(
        heads > 0 && hidden % heads == 0,
        "hidden must be divisible by heads"
    );
    BertShape {
        batch,
        seq,
        hidden,
        heads,
        ffn,
        layers,
    }
}

fn parse_state_mode(args: &[String]) -> StateMode {
    match arg_value(args, "--state-mode", Some("synthetic")).as_str() {
        "synthetic" => StateMode::Synthetic,
        "chained" => StateMode::Chained,
        other => {
            eprintln!("invalid --state-mode: {other}");
            usage();
        }
    }
}

fn parse_input_mode(args: &[String]) -> InputMode {
    match arg_value(args, "--input-mode", Some("synthetic")).as_str() {
        "synthetic" => InputMode::Synthetic,
        "real_io" => InputMode::RealIo,
        other => {
            eprintln!("invalid --input-mode: {other}");
            usage();
        }
    }
}

fn parse_rescale_bits(args: &[String]) -> u32 {
    arg_value(args, "--rescale-bits", Some("0"))
        .parse()
        .expect("invalid --rescale-bits")
}

fn parse_scale_bits(args: &[String]) -> u32 {
    arg_value(args, "--scale-bits", Some("16"))
        .parse()
        .expect("invalid --scale-bits")
}

fn parse_rescale_mode(args: &[String]) -> RescaleMode {
    match arg_value(args, "--rescale-mode", Some("local")).as_str() {
        "local" => RescaleMode::Local,
        "hp_clear" => RescaleMode::HpClear,
        other => {
            eprintln!("invalid --rescale-mode: {other}");
            usage();
        }
    }
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

fn share_real(i: usize, id: u8) -> f64 {
    let mut s = 0xBEE5_5000_1234_0000u64 ^ ((i as u64) << 5);
    let value = -4.0 + unit(&mut s) * 8.0;
    let share0 = -8.0 + unit(&mut s) * 16.0;
    if id == 0 {
        share0
    } else {
        value - share0
    }
}

fn share_real_rows(rows: usize, cols: usize, id: u8) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| share_real(r * cols + c, id)).collect())
        .collect()
}

fn left_share(m: usize, k: usize, seed: u64, id: u8) -> Vec<u64> {
    let a = make_vec(seed ^ 0xA001, m * k);
    share_vec(&a, seed ^ 0xC003, id)
}

fn weight_share(k: usize, n: usize, seed: u64, id: u8) -> Vec<u64> {
    let b = make_vec(seed ^ 0xB002, k * n);
    share_vec(&b, seed ^ 0xD004, id)
}

fn read_u64_file(path: &Path, expected_len: usize) -> std::io::Result<Vec<u64>> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let expected_bytes = expected_len * 8;
    if bytes.len() != expected_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "share file byte length mismatch: {} got={} expected={}",
                path.display(),
                bytes.len(),
                expected_bytes
            ),
        ));
    }
    let mut out = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        out.push(u64::from_le_bytes(buf));
    }
    Ok(out)
}

fn read_share_file(dir: &Path, name: &str, expected_len: usize) -> std::io::Result<Vec<u64>> {
    read_u64_file(&dir.join(name), expected_len)
}

fn read_share_file_with_fallback(
    dir: &Path,
    fallback_dir: Option<&Path>,
    name: &str,
    expected_len: usize,
) -> std::io::Result<Vec<u64>> {
    let primary = dir.join(name);
    if primary.exists() {
        return read_u64_file(&primary, expected_len);
    }
    if let Some(fallback) = fallback_dir {
        let fallback_path = fallback.join(name);
        if fallback_path.exists() {
            return read_cached_model_share(&fallback_path, expected_len);
        }
    }
    read_u64_file(&primary, expected_len)
}

fn read_cached_model_share(path: &Path, expected_len: usize) -> std::io::Result<Vec<u64>> {
    let cache = MODEL_SHARE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "model share cache poisoned"))?;
    if let Some(values) = guard.get(path) {
        assert_eq!(
            values.len(),
            expected_len,
            "cached model share length mismatch: {}",
            path.display()
        );
        return Ok(values.clone());
    }
    let values = read_u64_file(path, expected_len)?;
    guard.insert(path.to_path_buf(), values.clone());
    Ok(values)
}

fn read_optional_share_file(
    dir: &Path,
    name: &str,
    expected_len: usize,
) -> std::io::Result<Option<Vec<u64>>> {
    let path = dir.join(name);
    if path.exists() {
        read_u64_file(&path, expected_len).map(Some)
    } else {
        Ok(None)
    }
}

fn hidden_share(tokens: usize, hidden: usize, id: u8) -> Vec<u64> {
    let values = make_vec(0xC0DE_BE27_5000_0001u64, tokens * hidden);
    share_vec(&values, 0xC0DE_BE27_5000_0002u64, id)
}

fn add_share_vec(left: &[u64], right: &[u64]) -> Vec<u64> {
    assert_eq!(left.len(), right.len(), "residual add length mismatch");
    left.iter()
        .zip(right.iter())
        .map(|(&x, &y)| add(x, y))
        .collect()
}

fn add_bias_share(matrix: &mut [u64], bias: &[u64], rows: usize, cols: usize) {
    assert_eq!(matrix.len(), rows * cols, "bias add matrix shape mismatch");
    assert_eq!(bias.len(), cols, "bias add vector shape mismatch");
    for row in matrix.chunks_exact_mut(cols) {
        for (value, &bias_value) in row.iter_mut().zip(bias.iter()) {
            *value = add(*value, bias_value);
        }
    }
}

fn concat_weight_columns(weights: [&[u64]; 3], rows: usize, cols: usize) -> Vec<u64> {
    for weight in weights {
        assert_eq!(weight.len(), rows * cols, "qkv weight shape mismatch");
    }
    let mut out = Vec::with_capacity(rows * cols * 3);
    for row_idx in 0..rows {
        for weight in weights {
            out.extend_from_slice(&weight[row_idx * cols..(row_idx + 1) * cols]);
        }
    }
    out
}

fn split_qkv_projection(values: Vec<u64>, rows: usize, cols: usize) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    assert_eq!(values.len(), rows * cols * 3, "qkv projection shape mismatch");
    let mut q = Vec::with_capacity(rows * cols);
    let mut k = Vec::with_capacity(rows * cols);
    let mut v = Vec::with_capacity(rows * cols);
    for row in values.chunks_exact(cols * 3) {
        q.extend_from_slice(&row[..cols]);
        k.extend_from_slice(&row[cols..cols * 2]);
        v.extend_from_slice(&row[cols * 2..cols * 3]);
    }
    (q, k, v)
}

fn trunc_share(id: u8, share: u64, bits: u32) -> u64 {
    if bits == 0 {
        return share;
    }
    if id == 0 {
        share >> bits
    } else {
        (share.wrapping_neg() >> bits).wrapping_neg()
    }
}

fn rescale_share_vec(id: u8, shares: Vec<u64>, bits: u32) -> Vec<u64> {
    if bits == 0 {
        return shares;
    }
    shares
        .into_iter()
        .map(|share| trunc_share(id, share, bits))
        .collect()
}

fn signed_shift_share(value: u64, bits: u32) -> u64 {
    if bits == 0 {
        value
    } else {
        ((value as i64) >> bits) as u64
    }
}

fn fixed_ring_to_real(value: u64, scale_bits: u32) -> f64 {
    (value as i64) as f64 / ((1u64 << scale_bits) as f64)
}

fn real_to_fixed_ring(value: f64, scale_bits: u32) -> u64 {
    (value * ((1u64 << scale_bits) as f64)).round() as i64 as u64
}

fn party_fixed_to_real_hp_clear(
    comm: &SocketPartyEndpoint,
    role: &str,
    layer: usize,
    module: &str,
    shares: &[u64],
) -> Vec<f64> {
    let start = Instant::now();
    comm.send_to_hp(mcu_rust::channel::Msg::ShareVec(shares.to_vec()));
    let out = comm.recv_from_hp().into_real_vec();
    assert_eq!(out.len(), shares.len(), "fixed-to-real length mismatch");
    log_module(
        role,
        layer,
        &format!("{module}_to_real_hp_clear"),
        start.elapsed().as_secs_f64(),
        shares.len(),
    );
    out
}

fn apply_attention_scale_and_mask(
    id: u8,
    scores: &mut [f64],
    mask: &[u64],
    batch_idx: usize,
    shape: BertShape,
) {
    let inv_sqrt_head = 1.0 / ((shape.hidden / shape.heads) as f64).sqrt();
    for value in scores.iter_mut() {
        *value *= inv_sqrt_head;
    }
    for query_idx in 0..shape.seq {
        let row_offset = query_idx * shape.seq;
        for key_idx in 0..shape.seq {
            let keep = mask[batch_idx * shape.seq + key_idx] != 0;
            if !keep && id == 0 {
                scores[row_offset + key_idx] += -80.0;
            }
        }
    }
}

fn hp_fixed_to_real_hp_clear(
    prg: &mut PrgSync,
    comm: &SocketHpEndpoint,
    layer: usize,
    module: &str,
    len: usize,
    scale_bits: u32,
) {
    let start = Instant::now();
    let (msg0, msg1) = comm.recv_from_parties();
    let p0 = msg0.into_share_vec();
    let p1 = msg1.into_share_vec();
    assert_eq!(p0.len(), len, "p0 fixed-to-real length mismatch");
    assert_eq!(p1.len(), len, "p1 fixed-to-real length mismatch");
    let mut out0 = Vec::with_capacity(len);
    let mut out1 = Vec::with_capacity(len);
    for (&x0, &x1) in p0.iter().zip(&p1) {
        let value = fixed_ring_to_real(add(x0, x1), scale_bits);
        let s0 = prg.next_real(MOD);
        out0.push(s0);
        out1.push(value - s0);
    }
    comm.send_to_parties(
        mcu_rust::channel::Msg::RealVec(out0),
        mcu_rust::channel::Msg::RealVec(out1),
    );
    log_module(
        "hp",
        layer,
        &format!("{module}_to_real_hp_clear"),
        start.elapsed().as_secs_f64(),
        len,
    );
}

fn party_real_to_fixed_hp_clear(
    comm: &SocketPartyEndpoint,
    role: &str,
    layer: usize,
    module: &str,
    shares: &[f64],
) -> Vec<u64> {
    let start = Instant::now();
    comm.send_to_hp(mcu_rust::channel::Msg::RealVec(shares.to_vec()));
    let out = comm.recv_from_hp().into_share_vec();
    assert_eq!(out.len(), shares.len(), "real-to-fixed length mismatch");
    log_module(
        role,
        layer,
        &format!("{module}_to_ring_hp_clear"),
        start.elapsed().as_secs_f64(),
        shares.len(),
    );
    out
}

fn hp_real_to_fixed_hp_clear(
    prg: &mut PrgSync,
    comm: &SocketHpEndpoint,
    layer: usize,
    module: &str,
    len: usize,
    scale_bits: u32,
) {
    let start = Instant::now();
    let (msg0, msg1) = comm.recv_from_parties();
    let p0 = msg0.into_real_vec();
    let p1 = msg1.into_real_vec();
    assert_eq!(p0.len(), len, "p0 real-to-fixed length mismatch");
    assert_eq!(p1.len(), len, "p1 real-to-fixed length mismatch");
    let mut out0 = Vec::with_capacity(len);
    let mut out1 = Vec::with_capacity(len);
    for (&x0, &x1) in p0.iter().zip(&p1) {
        let value = real_to_fixed_ring(x0 + x1, scale_bits);
        let s0 = prg.next();
        out0.push(s0);
        out1.push(sub(value, s0));
    }
    comm.send_to_parties(
        mcu_rust::channel::Msg::ShareVec(out0),
        mcu_rust::channel::Msg::ShareVec(out1),
    );
    log_module(
        "hp",
        layer,
        &format!("{module}_to_ring_hp_clear"),
        start.elapsed().as_secs_f64(),
        len,
    );
}

fn party_tanh_hp_clear(
    comm: &SocketPartyEndpoint,
    role: &str,
    layer: usize,
    module: &str,
    shares: &[f64],
) -> Vec<f64> {
    let start = Instant::now();
    comm.send_to_hp(mcu_rust::channel::Msg::RealVec(shares.to_vec()));
    let out = comm.recv_from_hp().into_real_vec();
    assert_eq!(out.len(), shares.len(), "tanh output length mismatch");
    log_module(
        role,
        layer,
        &format!("{module}_tanh_hp_clear"),
        start.elapsed().as_secs_f64(),
        shares.len(),
    );
    out
}

fn hp_tanh_hp_clear(
    prg: &mut PrgSync,
    comm: &SocketHpEndpoint,
    layer: usize,
    module: &str,
    len: usize,
) {
    let start = Instant::now();
    let (msg0, msg1) = comm.recv_from_parties();
    let p0 = msg0.into_real_vec();
    let p1 = msg1.into_real_vec();
    assert_eq!(p0.len(), len, "p0 tanh length mismatch");
    assert_eq!(p1.len(), len, "p1 tanh length mismatch");
    let mut out0 = Vec::with_capacity(len);
    let mut out1 = Vec::with_capacity(len);
    for (&x0, &x1) in p0.iter().zip(&p1) {
        let value = (x0 + x1).tanh();
        let s0 = prg.next_real(MOD);
        out0.push(s0);
        out1.push(value - s0);
    }
    comm.send_to_parties(
        mcu_rust::channel::Msg::RealVec(out0),
        mcu_rust::channel::Msg::RealVec(out1),
    );
    log_module(
        "hp",
        layer,
        &format!("{module}_tanh_hp_clear"),
        start.elapsed().as_secs_f64(),
        len,
    );
}

fn party_reveal_logits_probs_hp_clear(
    comm: &SocketPartyEndpoint,
    role: &str,
    layer: usize,
    logits: &[u64],
    batch: usize,
    num_labels: usize,
) -> (Vec<f64>, Vec<f64>, usize) {
    let start = Instant::now();
    assert_eq!(logits.len(), batch * num_labels, "logits shape mismatch");
    comm.send_to_hp(mcu_rust::channel::Msg::ShareVec(logits.to_vec()));
    let (logits_real, probs) = comm.recv_from_hp().into_real_pair_vec();
    assert_eq!(
        logits_real.len(),
        logits.len(),
        "revealed logits length mismatch"
    );
    assert_eq!(probs.len(), logits.len(), "probability length mismatch");
    let prediction = probs[..num_labels]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    log_module(
        role,
        layer,
        "classifier_softmax_reveal_hp_clear",
        start.elapsed().as_secs_f64(),
        num_labels,
    );
    (logits_real, probs, prediction)
}

fn predictions_from_probs(probs: &[f64], num_labels: usize) -> Vec<usize> {
    probs
        .chunks_exact(num_labels)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

fn hp_reveal_logits_probs_hp_clear(
    comm: &SocketHpEndpoint,
    layer: usize,
    batch: usize,
    num_labels: usize,
    scale_bits: u32,
) {
    let start = Instant::now();
    let len = batch * num_labels;
    let (msg0, msg1) = comm.recv_from_parties();
    let p0 = msg0.into_share_vec();
    let p1 = msg1.into_share_vec();
    assert_eq!(p0.len(), len, "p0 logits length mismatch");
    assert_eq!(p1.len(), len, "p1 logits length mismatch");
    let logits: Vec<f64> = (0..len)
        .map(|idx| fixed_ring_to_real(add(p0[idx], p1[idx]), scale_bits))
        .collect();
    let mut probs = Vec::with_capacity(len);
    for row in logits.chunks_exact(num_labels) {
        let max_logit = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|value| (value - max_logit).exp()).collect();
        let denom = exps.iter().sum::<f64>();
        probs.extend(exps.iter().map(|value| value / denom));
    }
    comm.send_to_parties(
        mcu_rust::channel::Msg::RealPairVec {
            a: logits.clone(),
            b: probs.clone(),
        },
        mcu_rust::channel::Msg::RealPairVec {
            a: logits,
            b: probs,
        },
    );
    log_module(
        "hp",
        layer,
        "classifier_softmax_reveal_hp_clear",
        start.elapsed().as_secs_f64(),
        len,
    );
}

fn party_layer_norm_hp_clear(
    comm: &SocketPartyEndpoint,
    role: &str,
    layer: usize,
    module: &str,
    shares: Vec<u64>,
    gamma: &[u64],
    beta: &[u64],
) -> Vec<u64> {
    let start = Instant::now();
    let units = shares.len();
    comm.send_to_hp(mcu_rust::channel::Msg::ShareVec(shares));
    comm.send_to_hp(mcu_rust::channel::Msg::ShareVec(gamma.to_vec()));
    comm.send_to_hp(mcu_rust::channel::Msg::ShareVec(beta.to_vec()));
    let out = comm.recv_from_hp().into_share_vec();
    assert_eq!(out.len(), units, "layernorm output length mismatch");
    log_module(
        role,
        layer,
        &format!("{module}_hp_clear"),
        start.elapsed().as_secs_f64(),
        units,
    );
    out
}

fn hp_layer_norm_hp_clear(
    prg: &mut PrgSync,
    comm: &SocketHpEndpoint,
    layer: usize,
    module: &str,
    rows: usize,
    cols: usize,
    scale_bits: u32,
) {
    let start = Instant::now();
    let len = rows * cols;
    let (msg0, msg1) = comm.recv_from_parties();
    let p0 = msg0.into_share_vec();
    let p1 = msg1.into_share_vec();
    assert_eq!(p0.len(), len, "p0 layernorm length mismatch");
    assert_eq!(p1.len(), len, "p1 layernorm length mismatch");
    let (msg0, msg1) = comm.recv_from_parties();
    let gamma0 = msg0.into_share_vec();
    let gamma1 = msg1.into_share_vec();
    let (msg0, msg1) = comm.recv_from_parties();
    let beta0 = msg0.into_share_vec();
    let beta1 = msg1.into_share_vec();
    assert_eq!(gamma0.len(), cols, "p0 layernorm gamma shape mismatch");
    assert_eq!(gamma1.len(), cols, "p1 layernorm gamma shape mismatch");
    assert_eq!(beta0.len(), cols, "p0 layernorm beta shape mismatch");
    assert_eq!(beta1.len(), cols, "p1 layernorm beta shape mismatch");
    let gamma_real: Vec<f64> = (0..cols)
        .map(|idx| fixed_ring_to_real(add(gamma0[idx], gamma1[idx]), scale_bits))
        .collect();
    let beta_real: Vec<f64> = (0..cols)
        .map(|idx| fixed_ring_to_real(add(beta0[idx], beta1[idx]), scale_bits))
        .collect();
    let mut out0 = Vec::with_capacity(len);
    let mut out1 = Vec::with_capacity(len);
    for row_idx in 0..rows {
        let offset = row_idx * cols;
        let mut row = Vec::with_capacity(cols);
        for col_idx in 0..cols {
            row.push(fixed_ring_to_real(
                add(p0[offset + col_idx], p1[offset + col_idx]),
                scale_bits,
            ));
        }
        let mean = row.iter().sum::<f64>() / cols as f64;
        let var = row
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / cols as f64;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for col_idx in 0..cols {
            let normalized = (row[col_idx] - mean) * inv_std;
            let value = real_to_fixed_ring(
                normalized * gamma_real[col_idx] + beta_real[col_idx],
                scale_bits,
            );
            let s0 = prg.next();
            out0.push(s0);
            out1.push(sub(value, s0));
        }
    }
    comm.send_to_parties(
        mcu_rust::channel::Msg::ShareVec(out0),
        mcu_rust::channel::Msg::ShareVec(out1),
    );
    log_module(
        "hp",
        layer,
        &format!("{module}_hp_clear"),
        start.elapsed().as_secs_f64(),
        len,
    );
}

fn party_rescale_vec(
    id: u8,
    comm: &SocketPartyEndpoint,
    role: &str,
    layer: usize,
    module: &str,
    shares: Vec<u64>,
    bits: u32,
    mode: RescaleMode,
) -> Vec<u64> {
    if bits == 0 {
        return shares;
    }
    let start = Instant::now();
    let units = shares.len();
    let out = match mode {
        RescaleMode::Local => rescale_share_vec(id, shares, bits),
        RescaleMode::HpClear => {
            comm.send_to_hp(mcu_rust::channel::Msg::ShareVec(shares));
            comm.recv_from_hp().into_share_vec()
        }
    };
    log_module(
        role,
        layer,
        &format!("{module}_rescale_{}", mode.as_str()),
        start.elapsed().as_secs_f64(),
        units,
    );
    out
}

fn hp_rescale_vec(
    prg: &mut PrgSync,
    comm: &SocketHpEndpoint,
    layer: usize,
    module: &str,
    len: usize,
    bits: u32,
    mode: RescaleMode,
) {
    if bits == 0 || mode == RescaleMode::Local {
        return;
    }
    let start = Instant::now();
    match mode {
        RescaleMode::Local => {}
        RescaleMode::HpClear => {
            let (msg0, msg1) = comm.recv_from_parties();
            let p0 = msg0.into_share_vec();
            let p1 = msg1.into_share_vec();
            assert_eq!(p0.len(), len, "p0 rescale length mismatch");
            assert_eq!(p1.len(), len, "p1 rescale length mismatch");
            let mut out0 = Vec::with_capacity(len);
            let mut out1 = Vec::with_capacity(len);
            for (&x0, &x1) in p0.iter().zip(&p1) {
                let y = signed_shift_share(add(x0, x1), bits);
                let s0 = prg.next();
                out0.push(s0);
                out1.push(sub(y, s0));
            }
            comm.send_to_parties(
                mcu_rust::channel::Msg::ShareVec(out0),
                mcu_rust::channel::Msg::ShareVec(out1),
            );
        }
    }
    log_module(
        "hp",
        layer,
        &format!("{module}_rescale_{}", mode.as_str()),
        start.elapsed().as_secs_f64(),
        len,
    );
}

fn head_matrix(values: &[u64], batch_idx: usize, head_idx: usize, shape: BertShape) -> Vec<u64> {
    let head_dim = shape.hidden / shape.heads;
    let mut out = Vec::with_capacity(shape.seq * head_dim);
    for seq_idx in 0..shape.seq {
        let base = (batch_idx * shape.seq + seq_idx) * shape.hidden + head_idx * head_dim;
        out.extend_from_slice(&values[base..base + head_dim]);
    }
    out
}

fn head_matrix_transposed(
    values: &[u64],
    batch_idx: usize,
    head_idx: usize,
    shape: BertShape,
) -> Vec<u64> {
    let head_dim = shape.hidden / shape.heads;
    let mut out = Vec::with_capacity(head_dim * shape.seq);
    for dim_idx in 0..head_dim {
        for seq_idx in 0..shape.seq {
            let idx =
                (batch_idx * shape.seq + seq_idx) * shape.hidden + head_idx * head_dim + dim_idx;
            out.push(values[idx]);
        }
    }
    out
}

fn concat_head_contexts(contexts: &[Vec<u64>], shape: BertShape) -> Vec<u64> {
    let head_dim = shape.hidden / shape.heads;
    assert_eq!(
        contexts.len(),
        shape.batch * shape.heads,
        "attention context count mismatch"
    );
    let mut out = vec![0u64; shape.batch * shape.seq * shape.hidden];
    for batch_idx in 0..shape.batch {
        for head_idx in 0..shape.heads {
            let context = &contexts[batch_idx * shape.heads + head_idx];
            assert_eq!(
                context.len(),
                shape.seq * head_dim,
                "attention context shape mismatch"
            );
            for seq_idx in 0..shape.seq {
                for dim_idx in 0..head_dim {
                    let src = seq_idx * head_dim + dim_idx;
                    let dst = (batch_idx * shape.seq + seq_idx) * shape.hidden
                        + head_idx * head_dim
                        + dim_idx;
                    out[dst] = context[src];
                }
            }
        }
    }
    out
}

fn cls_rows(values: &[u64], shape: BertShape) -> Vec<u64> {
    let mut out = Vec::with_capacity(shape.batch * shape.hidden);
    for batch_idx in 0..shape.batch {
        let offset = batch_idx * shape.seq * shape.hidden;
        out.extend_from_slice(&values[offset..offset + shape.hidden]);
    }
    out
}

fn log_module(role: &str, layer: usize, module: &str, elapsed: f64, units: usize) {
    let rate = if elapsed > 0.0 {
        units as f64 / elapsed
    } else {
        0.0
    };
    println!(
        "[{role}] bert_module layer={layer} module={module} elapsed_s={elapsed:.9} units={units} units_per_s={rate:.3}"
    );
}

fn run_party_matmul(
    id: u8,
    prg: &mut PrgSync,
    comm: &SocketPartyEndpoint,
    layer: usize,
    module: &str,
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
) -> Vec<u64> {
    let a = left_share(m, k, seed, id);
    run_party_matmul_with_left(id, prg, comm, layer, module, &a, k, n, seed, m)
}

fn run_party_matmul_with_left(
    id: u8,
    prg: &mut PrgSync,
    comm: &SocketPartyEndpoint,
    layer: usize,
    module: &str,
    left: &[u64],
    k: usize,
    n: usize,
    seed: u64,
    m: usize,
) -> Vec<u64> {
    assert_eq!(left.len(), m * k, "left matrix shape mismatch for {module}");
    let b = weight_share(k, n, seed, id);
    let start = Instant::now();
    let out = party_matmul(id, left, &b, m, k, n, prg, comm);
    log_module(
        &format!("p{id}"),
        layer,
        module,
        start.elapsed().as_secs_f64(),
        m * k * n,
    );
    out
}

fn run_party_matmul_with_weight(
    id: u8,
    prg: &mut PrgSync,
    comm: &SocketPartyEndpoint,
    layer: usize,
    module: &str,
    left: &[u64],
    weight: &[u64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u64> {
    assert_eq!(left.len(), m * k, "left matrix shape mismatch for {module}");
    assert_eq!(
        weight.len(),
        k * n,
        "weight matrix shape mismatch for {module}"
    );
    let start = Instant::now();
    let out = party_matmul(id, left, weight, m, k, n, prg, comm);
    log_module(
        &format!("p{id}"),
        layer,
        module,
        start.elapsed().as_secs_f64(),
        m * k * n,
    );
    out
}

fn run_hp_matmul(
    prg: &mut PrgSync,
    comm: &SocketHpEndpoint,
    layer: usize,
    module: &str,
    m: usize,
    k: usize,
    n: usize,
) {
    let start = Instant::now();
    hp_matmul(m, k, n, prg, comm);
    log_module(
        "hp",
        layer,
        module,
        start.elapsed().as_secs_f64(),
        m * k * n,
    );
}

fn connect_party_with_retry(addr: &str, id: u8) -> std::io::Result<SocketPartyEndpoint> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match SocketPartyEndpoint::connect(addr, id) {
            Ok(comm) => return Ok(comm),
            Err(err) if Instant::now() < deadline => {
                if !matches!(
                    err.kind(),
                    std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::NotConnected
                ) {
                    return Err(err);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(err) => return Err(err),
        }
    }
}

fn run_party_session(args: &[String], id: u8) -> std::io::Result<()> {
    let shape = parse_shape(args);
    let state_mode = parse_state_mode(args);
    let input_mode = parse_input_mode(args);
    let scale_bits = parse_scale_bits(args);
    let rescale_bits = parse_rescale_bits(args);
    let rescale_mode = parse_rescale_mode(args);
    let addr = arg_value(args, "--addr", Some("127.0.0.1:9400"));
    let out_path = arg_value(
        args,
        "--out",
        Some(if id == 0 { "p0.bert" } else { "p1.bert" }),
    );
    let share_dir = PathBuf::from(arg_value(args, "--share-dir", Some(".")));
    let model_share_dir_arg = arg_value(args, "--model-share-dir", Some(""));
    let model_share_dir = if model_share_dir_arg.is_empty() {
        None
    } else {
        Some(PathBuf::from(model_share_dir_arg))
    };
    let model_share_dir_ref = model_share_dir.as_deref();
    if input_mode == InputMode::RealIo && state_mode != StateMode::Chained {
        eprintln!("real_io requires --state-mode chained");
        usage();
    }
    let comm = connect_party_with_retry(&addr, id)?;
    let mut prg = PrgSync::new(&SHARED_SEED);
    reset_socket_stats();
    reset_tensor_stats();
    let total_start = Instant::now();
    let tokens = shape.batch * shape.seq;
    let head_dim = shape.hidden / shape.heads;
    let attn_rows = shape.batch * shape.heads * shape.seq;
    let mut hidden = if input_mode == InputMode::RealIo {
        read_share_file(&share_dir, "hidden.bin", tokens * shape.hidden)?
    } else {
        hidden_share(tokens, shape.hidden, id)
    };
    let attention_mask = if input_mode == InputMode::RealIo {
        read_optional_share_file(&share_dir, "attention_mask.bin", tokens)?.unwrap_or_else(|| {
            eprintln!("[p{id}] attention_mask.bin not found; using all-ones mask");
            vec![1u64; tokens]
        })
    } else {
        vec![1u64; tokens]
    };

    println!(
        "[p{id}] bert_session_start state_mode={} input_mode={} scale_bits={} rescale_bits={} rescale_mode={} batch={} seq={} hidden={} heads={} ffn={} layers={}",
        state_mode.as_str(),
        input_mode.as_str(),
        scale_bits,
        rescale_bits,
        rescale_mode.as_str(),
        shape.batch,
        shape.seq,
        shape.hidden,
        shape.heads,
        shape.ffn,
        shape.layers
    );

    for layer in 0..shape.layers {
        if input_mode == InputMode::RealIo {
            let wq = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_Wq.bin"),
                shape.hidden * shape.hidden,
            )?;
            let bq = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_b_q.bin"),
                shape.hidden,
            )?;
            let wk = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_Wk.bin"),
                shape.hidden * shape.hidden,
            )?;
            let bk = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_b_k.bin"),
                shape.hidden,
            )?;
            let wv = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_Wv.bin"),
                shape.hidden * shape.hidden,
            )?;
            let bv = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_b_v.bin"),
                shape.hidden,
            )?;
            let wo = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_Wo.bin"),
                shape.hidden * shape.hidden,
            )?;
            let bo = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_b_o.bin"),
                shape.hidden,
            )?;
            let w1 = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_W1.bin"),
                shape.hidden * shape.ffn,
            )?;
            let b1 = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_b1.bin"),
                shape.ffn,
            )?;
            let w2 = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_W2.bin"),
                shape.ffn * shape.hidden,
            )?;
            let b2 = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_b2.bin"),
                shape.hidden,
            )?;
            let ln1_g = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_ln1_g.bin"),
                shape.hidden,
            )?;
            let ln1_b = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_ln1_b.bin"),
                shape.hidden,
            )?;
            let ln2_g = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_ln2_g.bin"),
                shape.hidden,
            )?;
            let ln2_b = read_share_file_with_fallback(
                &share_dir,
                model_share_dir_ref,
                &format!("layer_{layer:02}_ln2_b.bin"),
                shape.hidden,
            )?;

            let role = format!("p{id}");
            let qkv_weight = concat_weight_columns([&wq, &wk, &wv], shape.hidden, shape.hidden);
            let qkv = party_rescale_vec(
                id,
                &comm,
                &role,
                layer,
                "qkv_proj",
                run_party_matmul_with_weight(
                    id,
                    &mut prg,
                    &comm,
                    layer,
                    "qkv_proj",
                    &hidden,
                    &qkv_weight,
                    tokens,
                    shape.hidden,
                    shape.hidden * 3,
                ),
                rescale_bits,
                rescale_mode,
            );
            let (mut q, mut k, mut v) = split_qkv_projection(qkv, tokens, shape.hidden);
            add_bias_share(&mut q, &bq, tokens, shape.hidden);
            add_bias_share(&mut k, &bk, tokens, shape.hidden);
            add_bias_share(&mut v, &bv, tokens, shape.hidden);

            let mut score_shares = Vec::with_capacity(shape.batch * shape.heads);
            for batch_idx in 0..shape.batch {
                for head_idx in 0..shape.heads {
                    let q_head = head_matrix(&q, batch_idx, head_idx, shape);
                    let k_head_t = head_matrix_transposed(&k, batch_idx, head_idx, shape);
                    score_shares.push(party_rescale_vec(
                        id,
                        &comm,
                        &role,
                        layer,
                        "attn_scores",
                        run_party_matmul_with_weight(
                            id,
                            &mut prg,
                            &comm,
                            layer,
                            "attn_scores",
                            &q_head,
                            &k_head_t,
                            shape.seq,
                            head_dim,
                            shape.seq,
                        ),
                        rescale_bits,
                        rescale_mode,
                    ));
                }
            }

            let mut score_real_shares = Vec::with_capacity(attn_rows);
            for (idx, scores) in score_shares.iter().enumerate() {
                let batch_idx = idx / shape.heads;
                let mut score_matrix =
                    party_fixed_to_real_hp_clear(&comm, &role, layer, "attn_scores", scores);
                apply_attention_scale_and_mask(
                    id,
                    &mut score_matrix,
                    &attention_mask,
                    batch_idx,
                    shape,
                );
                for row in score_matrix.chunks_exact(shape.seq) {
                    score_real_shares.push(row.to_vec());
                }
            }
            let start = Instant::now();
            let softmax_real = party_softmax_batch(id, &score_real_shares, &mut prg, &comm);
            log_module(
                &format!("p{id}"),
                layer,
                "attn_softmax",
                start.elapsed().as_secs_f64(),
                attn_rows * shape.seq,
            );
            let mut softmax_ring = Vec::with_capacity(shape.batch * shape.heads);
            for head_idx in 0..shape.batch * shape.heads {
                let start_row = head_idx * shape.seq;
                let mut flat_probs = Vec::with_capacity(shape.seq * shape.seq);
                for row in &softmax_real[start_row..start_row + shape.seq] {
                    flat_probs.extend_from_slice(row);
                }
                softmax_ring.push(party_real_to_fixed_hp_clear(
                    &comm,
                    &role,
                    layer,
                    "attn_softmax",
                    &flat_probs,
                ));
            }

            let mut contexts = Vec::with_capacity(shape.batch * shape.heads);
            for (idx, probs) in softmax_ring.iter().enumerate() {
                let batch_idx = idx / shape.heads;
                let head_idx = idx % shape.heads;
                let v_head = head_matrix(&v, batch_idx, head_idx, shape);
                contexts.push(party_rescale_vec(
                    id,
                    &comm,
                    &role,
                    layer,
                    "attn_value",
                    run_party_matmul_with_weight(
                        id,
                        &mut prg,
                        &comm,
                        layer,
                        "attn_value",
                        probs,
                        &v_head,
                        shape.seq,
                        shape.seq,
                        head_dim,
                    ),
                    rescale_bits,
                    rescale_mode,
                ));
            }
            let attn_context = concat_head_contexts(&contexts, shape);
            let mut attn_out = party_rescale_vec(
                id,
                &comm,
                &role,
                layer,
                "o_proj",
                run_party_matmul_with_weight(
                    id,
                    &mut prg,
                    &comm,
                    layer,
                    "o_proj",
                    &attn_context,
                    &wo,
                    tokens,
                    shape.hidden,
                    shape.hidden,
                ),
                rescale_bits,
                rescale_mode,
            );
            add_bias_share(&mut attn_out, &bo, tokens, shape.hidden);
            hidden = add_share_vec(&hidden, &attn_out);
            hidden = party_layer_norm_hp_clear(&comm, &role, layer, "ln1", hidden, &ln1_g, &ln1_b);

            let mut ffn_mid = party_rescale_vec(
                id,
                &comm,
                &role,
                layer,
                "ffn_in",
                run_party_matmul_with_weight(
                    id,
                    &mut prg,
                    &comm,
                    layer,
                    "ffn_in",
                    &hidden,
                    &w1,
                    tokens,
                    shape.hidden,
                    shape.ffn,
                ),
                rescale_bits,
                rescale_mode,
            );
            add_bias_share(&mut ffn_mid, &b1, tokens, shape.ffn);

            let gelu_input =
                party_fixed_to_real_hp_clear(&comm, &role, layer, "ffn_gelu", &ffn_mid);
            let start = Instant::now();
            let gelu_real = party_gelu_batch(id, &gelu_input, &mut prg, &comm);
            log_module(
                &format!("p{id}"),
                layer,
                "ffn_gelu",
                start.elapsed().as_secs_f64(),
                tokens * shape.ffn,
            );
            let gelu_ring =
                party_real_to_fixed_hp_clear(&comm, &role, layer, "ffn_gelu", &gelu_real);

            let mut ffn_out = party_rescale_vec(
                id,
                &comm,
                &role,
                layer,
                "ffn_out",
                run_party_matmul_with_weight(
                    id,
                    &mut prg,
                    &comm,
                    layer,
                    "ffn_out",
                    &gelu_ring,
                    &w2,
                    tokens,
                    shape.ffn,
                    shape.hidden,
                ),
                rescale_bits,
                rescale_mode,
            );
            add_bias_share(&mut ffn_out, &b2, tokens, shape.hidden);
            hidden = add_share_vec(&hidden, &ffn_out);
            hidden = party_layer_norm_hp_clear(&comm, &role, layer, "ln2", hidden, &ln2_g, &ln2_b);
        } else if state_mode == StateMode::Chained {
            let _q = run_party_matmul_with_left(
                id,
                &mut prg,
                &comm,
                layer,
                "q_proj",
                &hidden,
                shape.hidden,
                shape.hidden,
                layer as u64 ^ 0x10,
                tokens,
            );
            let _k = run_party_matmul_with_left(
                id,
                &mut prg,
                &comm,
                layer,
                "k_proj",
                &hidden,
                shape.hidden,
                shape.hidden,
                layer as u64 ^ 0x11,
                tokens,
            );
            let _v = run_party_matmul_with_left(
                id,
                &mut prg,
                &comm,
                layer,
                "v_proj",
                &hidden,
                shape.hidden,
                shape.hidden,
                layer as u64 ^ 0x12,
                tokens,
            );

            run_party_matmul(
                id,
                &mut prg,
                &comm,
                layer,
                "attn_scores",
                attn_rows,
                head_dim,
                shape.seq,
                layer as u64 ^ 0x51,
            );

            let shares = share_real_rows(attn_rows, shape.seq, id);
            let start = Instant::now();
            let _softmax = party_softmax_batch(id, &shares, &mut prg, &comm);
            log_module(
                &format!("p{id}"),
                layer,
                "attn_softmax",
                start.elapsed().as_secs_f64(),
                attn_rows * shape.seq,
            );

            let attn_context = run_party_matmul(
                id,
                &mut prg,
                &comm,
                layer,
                "attn_value",
                attn_rows,
                shape.seq,
                head_dim,
                layer as u64 ^ 0x52,
            );
            let attn_out = run_party_matmul_with_left(
                id,
                &mut prg,
                &comm,
                layer,
                "o_proj",
                &attn_context,
                shape.hidden,
                shape.hidden,
                layer as u64 ^ 0x13,
                tokens,
            );
            hidden = add_share_vec(&hidden, &attn_out);

            let ffn_mid = run_party_matmul_with_left(
                id,
                &mut prg,
                &comm,
                layer,
                "ffn_in",
                &hidden,
                shape.hidden,
                shape.ffn,
                layer as u64 ^ 0x53,
                tokens,
            );

            let gelu_shares: Vec<f64> =
                (0..tokens * shape.ffn).map(|i| share_real(i, id)).collect();
            let start = Instant::now();
            let _gelu = party_gelu_batch(id, &gelu_shares, &mut prg, &comm);
            log_module(
                &format!("p{id}"),
                layer,
                "ffn_gelu",
                start.elapsed().as_secs_f64(),
                tokens * shape.ffn,
            );

            let ffn_out = run_party_matmul_with_left(
                id,
                &mut prg,
                &comm,
                layer,
                "ffn_out",
                &ffn_mid,
                shape.ffn,
                shape.hidden,
                layer as u64 ^ 0x54,
                tokens,
            );
            hidden = add_share_vec(&hidden, &ffn_out);
        } else {
            for module in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                run_party_matmul(
                    id,
                    &mut prg,
                    &comm,
                    layer,
                    module,
                    tokens,
                    shape.hidden,
                    shape.hidden,
                    layer as u64,
                );
            }
            run_party_matmul(
                id,
                &mut prg,
                &comm,
                layer,
                "attn_scores",
                attn_rows,
                head_dim,
                shape.seq,
                layer as u64 ^ 0x51,
            );

            let shares = share_real_rows(attn_rows, shape.seq, id);
            let start = Instant::now();
            let _softmax = party_softmax_batch(id, &shares, &mut prg, &comm);
            log_module(
                &format!("p{id}"),
                layer,
                "attn_softmax",
                start.elapsed().as_secs_f64(),
                attn_rows * shape.seq,
            );

            run_party_matmul(
                id,
                &mut prg,
                &comm,
                layer,
                "attn_value",
                attn_rows,
                shape.seq,
                head_dim,
                layer as u64 ^ 0x52,
            );
            run_party_matmul(
                id,
                &mut prg,
                &comm,
                layer,
                "ffn_in",
                tokens,
                shape.hidden,
                shape.ffn,
                layer as u64 ^ 0x53,
            );

            let gelu_shares: Vec<f64> =
                (0..tokens * shape.ffn).map(|i| share_real(i, id)).collect();
            let start = Instant::now();
            let _gelu = party_gelu_batch(id, &gelu_shares, &mut prg, &comm);
            log_module(
                &format!("p{id}"),
                layer,
                "ffn_gelu",
                start.elapsed().as_secs_f64(),
                tokens * shape.ffn,
            );

            run_party_matmul(
                id,
                &mut prg,
                &comm,
                layer,
                "ffn_out",
                tokens,
                shape.ffn,
                shape.hidden,
                layer as u64 ^ 0x54,
            );
        }
    }

    let mut final_logits = Vec::new();
    let mut final_probs = Vec::new();
    let mut final_prediction: Option<usize> = None;
    if input_mode == InputMode::RealIo {
        let role = format!("p{id}");
        let pooler_w = read_share_file_with_fallback(
            &share_dir,
            model_share_dir_ref,
            "pooler_W.bin",
            shape.hidden * shape.hidden,
        )?;
        let pooler_b =
            read_share_file_with_fallback(&share_dir, model_share_dir_ref, "pooler_b.bin", shape.hidden)?;
        let classifier_w = read_share_file_with_fallback(
            &share_dir,
            model_share_dir_ref,
            "classifier_W.bin",
            shape.hidden * 2,
        )?;
        let classifier_b =
            read_share_file_with_fallback(&share_dir, model_share_dir_ref, "classifier_b.bin", 2)?;
        let cls = cls_rows(&hidden, shape);
        let mut pooler_linear = party_rescale_vec(
            id,
            &comm,
            &role,
            shape.layers,
            "pooler_dense",
                run_party_matmul_with_weight(
                    id,
                    &mut prg,
                    &comm,
                shape.layers,
                "pooler_dense",
                &cls,
                &pooler_w,
                shape.batch,
                shape.hidden,
                shape.hidden,
            ),
            rescale_bits,
            rescale_mode,
        );
        add_bias_share(&mut pooler_linear, &pooler_b, shape.batch, shape.hidden);
        let pooler_real =
            party_fixed_to_real_hp_clear(&comm, &role, shape.layers, "pooler_tanh", &pooler_linear);
        let pooler_tanh = party_tanh_hp_clear(&comm, &role, shape.layers, "pooler", &pooler_real);
        let pooler_ring =
            party_real_to_fixed_hp_clear(&comm, &role, shape.layers, "pooler_tanh", &pooler_tanh);
        let mut logits = party_rescale_vec(
            id,
            &comm,
            &role,
            shape.layers,
            "classifier",
                run_party_matmul_with_weight(
                    id,
                    &mut prg,
                    &comm,
                shape.layers,
                "classifier",
                &pooler_ring,
                &classifier_w,
                shape.batch,
                shape.hidden,
                2,
            ),
            rescale_bits,
            rescale_mode,
        );
        add_bias_share(&mut logits, &classifier_b, shape.batch, 2);
        let (logits_real, probs, prediction) =
            party_reveal_logits_probs_hp_clear(&comm, &role, shape.layers, &logits, shape.batch, 2);
        final_logits = logits_real;
        final_probs = probs;
        final_prediction = Some(prediction);
    }

    let total = total_start.elapsed().as_secs_f64();
    print_timing(&format!("p{id}"), total);
    let mut writer = BufWriter::new(File::create(out_path)?);
    writeln!(writer, "role=p{id}")?;
    writeln!(writer, "status=ok")?;
    writeln!(writer, "state_mode={}", state_mode.as_str())?;
    writeln!(writer, "input_mode={}", input_mode.as_str())?;
    writeln!(writer, "rescale_bits={rescale_bits}")?;
    writeln!(writer, "rescale_mode={}", rescale_mode.as_str())?;
    writeln!(writer, "layers={}", shape.layers)?;
    if let Some(prediction) = final_prediction {
        let all_predictions = predictions_from_probs(&final_probs, 2);
        writeln!(writer, "prediction={prediction}")?;
        writeln!(
            writer,
            "prediction_label={}",
            if prediction == 0 {
                "negative"
            } else {
                "positive"
            }
        )?;
        writeln!(
            writer,
            "predictions={}",
            all_predictions
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )?;
        writeln!(
            writer,
            "prediction_labels={}",
            all_predictions
                .iter()
                .map(|&value| if value == 0 { "negative" } else { "positive" })
                .collect::<Vec<_>>()
                .join(",")
        )?;
        writeln!(
            writer,
            "logits={}",
            final_logits
                .iter()
                .map(|value| format!("{value:.12}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
        writeln!(
            writer,
            "probabilities={}",
            final_probs
                .iter()
                .map(|value| format!("{value:.12}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
    }
    writeln!(writer, "total_s={total:.9}")?;
    writer.flush()
}

fn run_hp_session(args: &[String]) -> std::io::Result<()> {
    let shape = parse_shape(args);
    let state_mode = parse_state_mode(args);
    let input_mode = parse_input_mode(args);
    let scale_bits = parse_scale_bits(args);
    let rescale_bits = parse_rescale_bits(args);
    let rescale_mode = parse_rescale_mode(args);
    if input_mode == InputMode::RealIo && state_mode != StateMode::Chained {
        eprintln!("real_io requires --state-mode chained");
        usage();
    }
    let addr = arg_value(args, "--addr", Some("127.0.0.1:9400"));
    println!(
        "[HP] listening on {addr}, kind=bert_session state_mode={} input_mode={} scale_bits={} rescale_bits={} rescale_mode={}",
        state_mode.as_str(),
        input_mode.as_str(),
        scale_bits,
        rescale_bits,
        rescale_mode.as_str()
    );
    let comm = SocketHpEndpoint::listen(&addr)?;
    println!("[HP] p0 and p1 connected");
    let mut prg = PrgSync::new(&HP_P0_SEED);
    reset_socket_stats();
    reset_tensor_stats();
    let total_start = Instant::now();
    let tokens = shape.batch * shape.seq;
    let head_dim = shape.hidden / shape.heads;
    let attn_rows = shape.batch * shape.heads * shape.seq;

    for layer in 0..shape.layers {
        if input_mode == InputMode::RealIo {
            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "qkv_proj",
                tokens,
                shape.hidden,
                shape.hidden * 3,
            );
            hp_rescale_vec(
                &mut prg,
                &comm,
                layer,
                "qkv_proj",
                tokens * shape.hidden * 3,
                rescale_bits,
                rescale_mode,
            );
            for _ in 0..shape.batch * shape.heads {
                run_hp_matmul(
                    &mut prg,
                    &comm,
                    layer,
                    "attn_scores",
                    shape.seq,
                    head_dim,
                    shape.seq,
                );
                hp_rescale_vec(
                    &mut prg,
                    &comm,
                    layer,
                    "attn_scores",
                    shape.seq * shape.seq,
                    rescale_bits,
                    rescale_mode,
                );
            }
            for _ in 0..shape.batch * shape.heads {
                hp_fixed_to_real_hp_clear(
                    &mut prg,
                    &comm,
                    layer,
                    "attn_scores",
                    shape.seq * shape.seq,
                    scale_bits,
                );
            }

            let start = Instant::now();
            hp_softmax_batch(attn_rows, shape.seq, &mut prg, &comm);
            log_module(
                "hp",
                layer,
                "attn_softmax",
                start.elapsed().as_secs_f64(),
                attn_rows * shape.seq,
            );
            for _ in 0..shape.batch * shape.heads {
                hp_real_to_fixed_hp_clear(
                    &mut prg,
                    &comm,
                    layer,
                    "attn_softmax",
                    shape.seq * shape.seq,
                    scale_bits,
                );
            }

            for _ in 0..shape.batch * shape.heads {
                run_hp_matmul(
                    &mut prg,
                    &comm,
                    layer,
                    "attn_value",
                    shape.seq,
                    shape.seq,
                    head_dim,
                );
                hp_rescale_vec(
                    &mut prg,
                    &comm,
                    layer,
                    "attn_value",
                    shape.seq * head_dim,
                    rescale_bits,
                    rescale_mode,
                );
            }
            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "o_proj",
                tokens,
                shape.hidden,
                shape.hidden,
            );
            hp_rescale_vec(
                &mut prg,
                &comm,
                layer,
                "o_proj",
                tokens * shape.hidden,
                rescale_bits,
                rescale_mode,
            );
            hp_layer_norm_hp_clear(
                &mut prg,
                &comm,
                layer,
                "ln1",
                tokens,
                shape.hidden,
                scale_bits,
            );
            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "ffn_in",
                tokens,
                shape.hidden,
                shape.ffn,
            );
            hp_rescale_vec(
                &mut prg,
                &comm,
                layer,
                "ffn_in",
                tokens * shape.ffn,
                rescale_bits,
                rescale_mode,
            );

            hp_fixed_to_real_hp_clear(
                &mut prg,
                &comm,
                layer,
                "ffn_gelu",
                tokens * shape.ffn,
                scale_bits,
            );
            let start = Instant::now();
            hp_gelu_batch(tokens * shape.ffn, &mut prg, &comm);
            log_module(
                "hp",
                layer,
                "ffn_gelu",
                start.elapsed().as_secs_f64(),
                tokens * shape.ffn,
            );
            hp_real_to_fixed_hp_clear(
                &mut prg,
                &comm,
                layer,
                "ffn_gelu",
                tokens * shape.ffn,
                scale_bits,
            );

            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "ffn_out",
                tokens,
                shape.ffn,
                shape.hidden,
            );
            hp_rescale_vec(
                &mut prg,
                &comm,
                layer,
                "ffn_out",
                tokens * shape.hidden,
                rescale_bits,
                rescale_mode,
            );
            hp_layer_norm_hp_clear(
                &mut prg,
                &comm,
                layer,
                "ln2",
                tokens,
                shape.hidden,
                scale_bits,
            );
        } else if state_mode == StateMode::Chained {
            for module in ["q_proj", "k_proj", "v_proj"] {
                run_hp_matmul(
                    &mut prg,
                    &comm,
                    layer,
                    module,
                    tokens,
                    shape.hidden,
                    shape.hidden,
                );
            }
            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "attn_scores",
                attn_rows,
                head_dim,
                shape.seq,
            );

            let start = Instant::now();
            hp_softmax_batch(attn_rows, shape.seq, &mut prg, &comm);
            log_module(
                "hp",
                layer,
                "attn_softmax",
                start.elapsed().as_secs_f64(),
                attn_rows * shape.seq,
            );

            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "attn_value",
                attn_rows,
                shape.seq,
                head_dim,
            );
            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "o_proj",
                tokens,
                shape.hidden,
                shape.hidden,
            );
            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "ffn_in",
                tokens,
                shape.hidden,
                shape.ffn,
            );

            let start = Instant::now();
            hp_gelu_batch(tokens * shape.ffn, &mut prg, &comm);
            log_module(
                "hp",
                layer,
                "ffn_gelu",
                start.elapsed().as_secs_f64(),
                tokens * shape.ffn,
            );

            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "ffn_out",
                tokens,
                shape.ffn,
                shape.hidden,
            );
        } else {
            for module in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                run_hp_matmul(
                    &mut prg,
                    &comm,
                    layer,
                    module,
                    tokens,
                    shape.hidden,
                    shape.hidden,
                );
            }
            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "attn_scores",
                attn_rows,
                head_dim,
                shape.seq,
            );

            let start = Instant::now();
            hp_softmax_batch(attn_rows, shape.seq, &mut prg, &comm);
            log_module(
                "hp",
                layer,
                "attn_softmax",
                start.elapsed().as_secs_f64(),
                attn_rows * shape.seq,
            );

            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "attn_value",
                attn_rows,
                shape.seq,
                head_dim,
            );
            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "ffn_in",
                tokens,
                shape.hidden,
                shape.ffn,
            );

            let start = Instant::now();
            hp_gelu_batch(tokens * shape.ffn, &mut prg, &comm);
            log_module(
                "hp",
                layer,
                "ffn_gelu",
                start.elapsed().as_secs_f64(),
                tokens * shape.ffn,
            );

            run_hp_matmul(
                &mut prg,
                &comm,
                layer,
                "ffn_out",
                tokens,
                shape.ffn,
                shape.hidden,
            );
        }
    }

    if input_mode == InputMode::RealIo {
        run_hp_matmul(
            &mut prg,
            &comm,
            shape.layers,
            "pooler_dense",
            shape.batch,
            shape.hidden,
            shape.hidden,
        );
        hp_rescale_vec(
            &mut prg,
            &comm,
            shape.layers,
            "pooler_dense",
            shape.batch * shape.hidden,
            rescale_bits,
            rescale_mode,
        );
        hp_fixed_to_real_hp_clear(
            &mut prg,
            &comm,
            shape.layers,
            "pooler_tanh",
            shape.batch * shape.hidden,
            scale_bits,
        );
        hp_tanh_hp_clear(
            &mut prg,
            &comm,
            shape.layers,
            "pooler",
            shape.batch * shape.hidden,
        );
        hp_real_to_fixed_hp_clear(
            &mut prg,
            &comm,
            shape.layers,
            "pooler_tanh",
            shape.batch * shape.hidden,
            scale_bits,
        );
        run_hp_matmul(
            &mut prg,
            &comm,
            shape.layers,
            "classifier",
            shape.batch,
            shape.hidden,
            2,
        );
        hp_rescale_vec(
            &mut prg,
            &comm,
            shape.layers,
            "classifier",
            shape.batch * 2,
            rescale_bits,
            rescale_mode,
        );
        hp_reveal_logits_probs_hp_clear(&comm, shape.layers, shape.batch, 2, scale_bits);
    }

    let total = total_start.elapsed().as_secs_f64();
    println!("[HP] done: {total:.6}s kind=bert_session");
    print_timing("hp", total);
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
    let local_s = (total_s - comm_s).max(0.0);
    println!(
        "[{role}] timing total_s={total_s:.9} comm_s={comm_s:.9} local_s={local_s:.9} compute_s={compute_s:.9} send_s={send_s:.9} recv_s={recv_s:.9} recv_wait_s={recv_wait_s:.9} recv_read_s={recv_read_s:.9} send_msgs={} recv_msgs={} send_bytes={} recv_bytes={} matmul_calls={} cpu_matmul_calls={} cuda_matmul_calls={} fused_party_calls={} fused_hp_calls={} cuda_fallbacks={}",
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

fn next_request(service_dir: &Path, seen: &[String]) -> std::io::Result<Option<String>> {
    let req_dir = service_dir.join("requests");
    fs::create_dir_all(&req_dir)?;
    let mut ids = Vec::new();
    for entry in fs::read_dir(req_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) == Some("req") {
            if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
                if !seen.iter().any(|item| item == stem) {
                    ids.push(stem.to_string());
                }
            }
        }
    }
    ids.sort();
    Ok(ids.into_iter().next())
}

fn without_options(args: &[String], names: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        if names.iter().any(|name| args[idx] == *name) {
            idx += 2;
        } else {
            out.push(args[idx].clone());
            idx += 1;
        }
    }
    out
}

fn service_args(args: &[String], request_id: &str, role: &str) -> Vec<String> {
    let mut out = without_options(
        args,
        &["--share-dir", "--model-share-dir", "--out", "--service-dir", "--out-dir"],
    );
    let service_dir = PathBuf::from(arg_value(args, "--service-dir", Some("/workspace/out/mcu_service")));
    let base_share_dir = PathBuf::from(arg_value(args, "--share-dir", Some("/workspace/bert_shares")));
    let base_model_share_dir = PathBuf::from(arg_value(args, "--model-share-dir", Some("")));
    let base_out_dir = PathBuf::from(arg_value(args, "--out-dir", Some("/workspace/out/mcu_service/responses")));
    out.push("--share-dir".to_string());
    out.push(
        base_share_dir
            .join(request_id)
            .join(role)
            .to_string_lossy()
            .to_string(),
    );
    if !base_model_share_dir.as_os_str().is_empty() {
        out.push("--model-share-dir".to_string());
        out.push(base_model_share_dir.join(role).to_string_lossy().to_string());
    }
    out.push("--out".to_string());
    out.push(
        base_out_dir
            .join(request_id)
            .join(format!("{role}.out"))
            .to_string_lossy()
            .to_string(),
    );
    out.push("--service-dir".to_string());
    out.push(service_dir.to_string_lossy().to_string());
    out
}

fn run_service_loop(args: &[String], role: &str) -> std::io::Result<()> {
    let service_dir = PathBuf::from(arg_value(args, "--service-dir", Some("/workspace/out/mcu_service")));
    let ready_path = service_dir.join(format!("{role}.ready"));
    let heartbeat_path = service_dir.join(format!("{role}.heartbeat"));
    let stop_path = service_dir.join("stop");
    let response_dir = service_dir.join("responses");
    fs::create_dir_all(service_dir.join("requests"))?;
    fs::create_dir_all(&response_dir)?;
    write_text_atomic(&ready_path, "ready\n")?;
    println!("[{role}] mcu_service_ready dir={}", service_dir.display());
    let mut seen = Vec::new();
    while !stop_path.exists() {
        write_text_atomic(&heartbeat_path, &format!("{:.6}\n", Instant::now().elapsed().as_secs_f64()))?;
        if let Some(request_id) = next_request(&service_dir, &seen)? {
            seen.push(request_id.clone());
            let req_start = Instant::now();
            let run_args = service_args(args, &request_id, role);
            let result = match role {
                "hp" => run_hp_session(&run_args),
                "p0" => run_party_session(&run_args, 0),
                "p1" => run_party_session(&run_args, 1),
                _ => unreachable!(),
            };
            let done_dir = response_dir.join(&request_id);
            fs::create_dir_all(&done_dir)?;
            match result {
                Ok(()) => {
                    write_text_atomic(
                        &done_dir.join(format!("{role}.done")),
                        &format!("elapsed_s={:.9}\n", req_start.elapsed().as_secs_f64()),
                    )?;
                    println!(
                        "[{role}] mcu_service_request_done id={} elapsed_s={:.9}",
                        request_id,
                        req_start.elapsed().as_secs_f64()
                    );
                }
                Err(err) => {
                    write_text_atomic(&done_dir.join(format!("{role}.error")), &format!("{err}\n"))?;
                    return Err(err);
                }
            }
        } else {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    let _ = fs::remove_file(ready_path);
    let _ = fs::remove_file(heartbeat_path);
    println!("[{role}] mcu_service_stopped");
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
    }
    match args[1].as_str() {
        "hp" => run_hp_session(&args[2..]),
        "p0" => run_party_session(&args[2..], 0),
        "p1" => run_party_session(&args[2..], 1),
        "service-hp" => run_service_loop(&args[2..], "hp"),
        "service-p0" => run_service_loop(&args[2..], "p0"),
        "service-p1" => run_service_loop(&args[2..], "p1"),
        _ => usage(),
    }
}
