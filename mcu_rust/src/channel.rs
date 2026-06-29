//! 通信抽象：`Comm` trait + 进程内 mock（std::sync::mpsc + 线程）。
//!
//! 结构对齐 Python 的三方消息流（每个发送方→HP 用独立信箱，HP→各方各一条），
//! 便于未来替换为真实 socket 实现。PyO3 高速路径不走该层（见 `simulate.rs`），
//! 此处用于忠实复现协议轮次与做结构性验证。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

/// 三方之间传递的消息（覆盖当前需通过 Comm 层演示的协议）。
#[derive(Debug, Clone)]
pub enum Msg {
    /// 乘法协议：P_i → HP 的掩码值。
    MulToHp { id: u8, mx: u64, my: u64 },
    /// Batched multiplication protocol: P_i -> HP masked x/y vectors.
    MulVecToHp { id: u8, mx: Vec<u64>, my: Vec<u64> },
    /// Matrix multiplication protocol: P_i → HP masked matrix blocks.
    MatMulToHp { id: u8, a: Vec<u64>, b: Vec<u64> },
    /// HP → P_i 的整数环份额。
    Share(u64),
    /// HP → P_i vector of integer-ring shares.
    ShareVec(Vec<u64>),
    /// Real-valued single payload.
    Real(f64),
    /// Real-valued vector payload.
    RealVec(Vec<f64>),
    /// Real-valued pair payload.
    RealPair(f64, f64),
    /// Real-valued pair vector payload.
    RealPairVec { a: Vec<f64>, b: Vec<f64> },
    /// Public bit broadcast.
    Bit(u8),
    /// Public bit vector broadcast.
    BitVec(Vec<u8>),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SocketStats {
    pub send_messages: u64,
    pub recv_messages: u64,
    pub send_bytes: u64,
    pub recv_bytes: u64,
    pub send_nanos: u64,
    pub recv_nanos: u64,
    pub recv_wait_nanos: u64,
    pub recv_read_nanos: u64,
}

static SEND_MESSAGES: AtomicU64 = AtomicU64::new(0);
static RECV_MESSAGES: AtomicU64 = AtomicU64::new(0);
static SEND_BYTES: AtomicU64 = AtomicU64::new(0);
static RECV_BYTES: AtomicU64 = AtomicU64::new(0);
static SEND_NANOS: AtomicU64 = AtomicU64::new(0);
static RECV_NANOS: AtomicU64 = AtomicU64::new(0);
static RECV_WAIT_NANOS: AtomicU64 = AtomicU64::new(0);
static RECV_READ_NANOS: AtomicU64 = AtomicU64::new(0);

pub fn reset_socket_stats() {
    SEND_MESSAGES.store(0, Ordering::Relaxed);
    RECV_MESSAGES.store(0, Ordering::Relaxed);
    SEND_BYTES.store(0, Ordering::Relaxed);
    RECV_BYTES.store(0, Ordering::Relaxed);
    SEND_NANOS.store(0, Ordering::Relaxed);
    RECV_NANOS.store(0, Ordering::Relaxed);
    RECV_WAIT_NANOS.store(0, Ordering::Relaxed);
    RECV_READ_NANOS.store(0, Ordering::Relaxed);
}

pub fn socket_stats_snapshot() -> SocketStats {
    SocketStats {
        send_messages: SEND_MESSAGES.load(Ordering::Relaxed),
        recv_messages: RECV_MESSAGES.load(Ordering::Relaxed),
        send_bytes: SEND_BYTES.load(Ordering::Relaxed),
        recv_bytes: RECV_BYTES.load(Ordering::Relaxed),
        send_nanos: SEND_NANOS.load(Ordering::Relaxed),
        recv_nanos: RECV_NANOS.load(Ordering::Relaxed),
        recv_wait_nanos: RECV_WAIT_NANOS.load(Ordering::Relaxed),
        recv_read_nanos: RECV_READ_NANOS.load(Ordering::Relaxed),
    }
}

impl Msg {
    pub fn as_share(&self) -> u64 {
        match self {
            Msg::Share(v) => *v,
            _ => panic!("期望 Share 消息"),
        }
    }
    pub fn as_mul(&self) -> (u8, u64, u64) {
        match self {
            Msg::MulToHp { id, mx, my } => (*id, *mx, *my),
            _ => panic!("期望 MulToHp 消息"),
        }
    }
    pub fn as_mul_vec(&self) -> (u8, &[u64], &[u64]) {
        match self {
            Msg::MulVecToHp { id, mx, my } => (*id, mx, my),
            _ => panic!("expected MulVecToHp message"),
        }
    }
    pub fn as_matmul(&self) -> (u8, &[u64], &[u64]) {
        match self {
            Msg::MatMulToHp { id, a, b } => (*id, a, b),
            _ => panic!("expected MatMulToHp message"),
        }
    }
    pub fn into_share_vec(self) -> Vec<u64> {
        match self {
            Msg::ShareVec(values) => values,
            _ => panic!("expected ShareVec message"),
        }
    }
    pub fn as_real(&self) -> f64 {
        match self {
            Msg::Real(v) => *v,
            _ => panic!("expected Real message"),
        }
    }
    pub fn into_real_vec(self) -> Vec<f64> {
        match self {
            Msg::RealVec(values) => values,
            _ => panic!("expected RealVec message"),
        }
    }
    pub fn as_real_pair(&self) -> (f64, f64) {
        match self {
            Msg::RealPair(a, b) => (*a, *b),
            _ => panic!("expected RealPair message"),
        }
    }
    pub fn into_real_pair_vec(self) -> (Vec<f64>, Vec<f64>) {
        match self {
            Msg::RealPairVec { a, b } => (a, b),
            _ => panic!("expected RealPairVec message"),
        }
    }
    pub fn as_bit(&self) -> u8 {
        match self {
            Msg::Bit(v) => *v,
            _ => panic!("expected Bit message"),
        }
    }
    pub fn into_bit_vec(self) -> Vec<u8> {
        match self {
            Msg::BitVec(values) => values,
            _ => panic!("expected BitVec message"),
        }
    }
}

/// 参与方（P0 / P1）视角的通信能力。
pub trait PartyComm {
    fn send_to_hp(&self, m: Msg);
    fn recv_from_hp(&self) -> Msg;
}

/// HP 视角的通信能力。
pub trait HpComm {
    fn recv_from_p0(&self) -> Msg;
    fn recv_from_p1(&self) -> Msg;
    fn recv_from_parties(&self) -> (Msg, Msg) {
        (self.recv_from_p0(), self.recv_from_p1())
    }
    fn send_to_p0(&self, m: Msg);
    fn send_to_p1(&self, m: Msg);
    fn send_to_parties(&self, p0: Msg, p1: Msg) {
        self.send_to_p0(p0);
        self.send_to_p1(p1);
    }
}

pub struct PartyEndpoint {
    to_hp: Sender<Msg>,
    from_hp: Receiver<Msg>,
}

pub struct HpEndpoint {
    from_p0: Receiver<Msg>,
    from_p1: Receiver<Msg>,
    to_p0: Sender<Msg>,
    to_p1: Sender<Msg>,
}

impl PartyComm for PartyEndpoint {
    fn send_to_hp(&self, m: Msg) {
        self.to_hp.send(m).expect("send_to_hp 失败");
    }
    fn recv_from_hp(&self) -> Msg {
        self.from_hp.recv().expect("recv_from_hp 失败")
    }
}

impl HpComm for HpEndpoint {
    fn recv_from_p0(&self) -> Msg {
        self.from_p0.recv().expect("recv_from_p0 失败")
    }
    fn recv_from_p1(&self) -> Msg {
        self.from_p1.recv().expect("recv_from_p1 失败")
    }
    fn send_to_p0(&self, m: Msg) {
        self.to_p0.send(m).expect("send_to_p0 失败");
    }
    fn send_to_p1(&self, m: Msg) {
        self.to_p1.send(m).expect("send_to_p1 失败");
    }
}

/// 构造一组三方 mock 通信端点。
pub fn make_mock() -> (PartyEndpoint, PartyEndpoint, HpEndpoint) {
    let (hp_from_p0_tx, hp_from_p0_rx) = channel();
    let (hp_from_p1_tx, hp_from_p1_rx) = channel();
    let (p0_tx, p0_rx) = channel();
    let (p1_tx, p1_rx) = channel();

    let p0 = PartyEndpoint {
        to_hp: hp_from_p0_tx,
        from_hp: p0_rx,
    };
    let p1 = PartyEndpoint {
        to_hp: hp_from_p1_tx,
        from_hp: p1_rx,
    };
    let hp = HpEndpoint {
        from_p0: hp_from_p0_rx,
        from_p1: hp_from_p1_rx,
        to_p0: p0_tx,
        to_p1: p1_tx,
    };
    (p0, p1, hp)
}

const MSG_MUL_TO_HP: u8 = 1;
const MSG_SHARE: u8 = 2;
const MSG_REAL: u8 = 3;
const MSG_REAL_PAIR: u8 = 4;
const MSG_BIT: u8 = 5;
const MSG_MATMUL_TO_HP: u8 = 6;
const MSG_SHARE_VEC: u8 = 7;
const MSG_MUL_VEC_TO_HP: u8 = 8;
const MSG_REAL_VEC: u8 = 9;
const MSG_REAL_PAIR_VEC: u8 = 10;
const MSG_BIT_VEC: u8 = 11;

fn u64_bytes(values: &[u64]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

fn f64_bytes(values: &[f64]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

fn write_slices(stream: &mut TcpStream, slices: &[&[u8]]) -> std::io::Result<()> {
    for slice in slices {
        stream.write_all(slice)?;
    }
    Ok(())
}

fn read_u64_vec(stream: &mut TcpStream) -> std::io::Result<Vec<u64>> {
    let mut len_buf = [0u8; 8];
    stream.read_exact(&mut len_buf)?;
    let len = u64::from_be_bytes(len_buf) as usize;
    let mut out = vec![0u64; len];
    let bytes = unsafe { slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, len * 8) };
    stream.read_exact(bytes)?;
    Ok(out)
}

fn read_f64_vec(stream: &mut TcpStream) -> std::io::Result<Vec<f64>> {
    let mut len_buf = [0u8; 8];
    stream.read_exact(&mut len_buf)?;
    let len = u64::from_be_bytes(len_buf) as usize;
    let mut out = vec![0.0f64; len];
    let bytes = unsafe { slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, len * 8) };
    stream.read_exact(bytes)?;
    Ok(out)
}

fn write_u8_vec(buf: &mut Vec<u8>, values: &[u8]) {
    buf.extend_from_slice(&(values.len() as u64).to_be_bytes());
    buf.extend_from_slice(values);
}

fn read_u8_vec(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 8];
    stream.read_exact(&mut len_buf)?;
    let len = u64::from_be_bytes(len_buf) as usize;
    let mut out = vec![0u8; len];
    stream.read_exact(&mut out)?;
    Ok(out)
}

fn write_msg(stream: &mut TcpStream, msg: &Msg) -> std::io::Result<()> {
    match msg {
        Msg::MulToHp { id, mx, my } => {
            let mut buf = [0u8; 18];
            buf[0] = MSG_MUL_TO_HP;
            buf[1] = *id;
            buf[2..10].copy_from_slice(&mx.to_be_bytes());
            buf[10..18].copy_from_slice(&my.to_be_bytes());
            stream.write_all(&buf)
        }
        Msg::MatMulToHp { id, a, b } => {
            let header = [MSG_MATMUL_TO_HP, *id];
            let a_len = (a.len() as u64).to_be_bytes();
            let b_len = (b.len() as u64).to_be_bytes();
            write_slices(stream, &[&header, &a_len, u64_bytes(a), &b_len, u64_bytes(b)])
        }
        Msg::MulVecToHp { id, mx, my } => {
            let header = [MSG_MUL_VEC_TO_HP, *id];
            let mx_len = (mx.len() as u64).to_be_bytes();
            let my_len = (my.len() as u64).to_be_bytes();
            write_slices(stream, &[&header, &mx_len, u64_bytes(mx), &my_len, u64_bytes(my)])
        }
        Msg::Share(v) => {
            let mut buf = [0u8; 9];
            buf[0] = MSG_SHARE;
            buf[1..9].copy_from_slice(&v.to_be_bytes());
            stream.write_all(&buf)
        }
        Msg::ShareVec(values) => {
            let tag = [MSG_SHARE_VEC];
            let len = (values.len() as u64).to_be_bytes();
            write_slices(stream, &[&tag, &len, u64_bytes(values)])
        }
        Msg::Real(v) => {
            let mut buf = [0u8; 9];
            buf[0] = MSG_REAL;
            buf[1..9].copy_from_slice(&v.to_bits().to_be_bytes());
            stream.write_all(&buf)
        }
        Msg::RealVec(values) => {
            let tag = [MSG_REAL_VEC];
            let len = (values.len() as u64).to_be_bytes();
            write_slices(stream, &[&tag, &len, f64_bytes(values)])
        }
        Msg::RealPair(a, b) => {
            let mut buf = [0u8; 17];
            buf[0] = MSG_REAL_PAIR;
            buf[1..9].copy_from_slice(&a.to_bits().to_be_bytes());
            buf[9..17].copy_from_slice(&b.to_bits().to_be_bytes());
            stream.write_all(&buf)
        }
        Msg::RealPairVec { a, b } => {
            let tag = [MSG_REAL_PAIR_VEC];
            let a_len = (a.len() as u64).to_be_bytes();
            let b_len = (b.len() as u64).to_be_bytes();
            write_slices(stream, &[&tag, &a_len, f64_bytes(a), &b_len, f64_bytes(b)])
        }
        Msg::Bit(v) => stream.write_all(&[MSG_BIT, *v]),
        Msg::BitVec(values) => {
            let mut buf = Vec::with_capacity(9 + values.len());
            buf.push(MSG_BIT_VEC);
            write_u8_vec(&mut buf, values);
            stream.write_all(&buf)
        }
    }
}

fn msg_wire_len(msg: &Msg) -> u64 {
    match msg {
        Msg::MulToHp { .. } => 18,
        Msg::MulVecToHp { mx, my, .. } => 18 + ((mx.len() + my.len()) as u64) * 8,
        Msg::MatMulToHp { a, b, .. } => 18 + ((a.len() + b.len()) as u64) * 8,
        Msg::Share(_) => 9,
        Msg::ShareVec(values) => 9 + (values.len() as u64) * 8,
        Msg::Real(_) => 9,
        Msg::RealVec(values) => 9 + (values.len() as u64) * 8,
        Msg::RealPair(_, _) => 17,
        Msg::RealPairVec { a, b } => 17 + ((a.len() + b.len()) as u64) * 8,
        Msg::Bit(_) => 2,
        Msg::BitVec(values) => 9 + values.len() as u64,
    }
}

fn record_send(elapsed_nanos: u64, bytes: u64) {
    SEND_MESSAGES.fetch_add(1, Ordering::Relaxed);
    SEND_BYTES.fetch_add(bytes, Ordering::Relaxed);
    SEND_NANOS.fetch_add(elapsed_nanos, Ordering::Relaxed);
}

fn record_recv(elapsed_nanos: u64, wait_nanos: u64, read_nanos: u64, bytes: u64) {
    record_recv_batch(1, elapsed_nanos, wait_nanos, read_nanos, bytes);
}

fn record_recv_batch(
    messages: u64,
    elapsed_nanos: u64,
    wait_nanos: u64,
    read_nanos: u64,
    bytes: u64,
) {
    RECV_MESSAGES.fetch_add(messages, Ordering::Relaxed);
    RECV_BYTES.fetch_add(bytes, Ordering::Relaxed);
    RECV_NANOS.fetch_add(elapsed_nanos, Ordering::Relaxed);
    RECV_WAIT_NANOS.fetch_add(wait_nanos, Ordering::Relaxed);
    RECV_READ_NANOS.fetch_add(read_nanos, Ordering::Relaxed);
}

fn read_msg(stream: &mut TcpStream) -> std::io::Result<(Msg, u64, u64)> {
    let wait_start = Instant::now();
    let mut tag = [0u8; 1];
    stream.read_exact(&mut tag)?;
    let wait_nanos = wait_start.elapsed().as_nanos() as u64;
    let read_start = Instant::now();
    let msg = match tag[0] {
        MSG_MUL_TO_HP => {
            let mut buf = [0u8; 17];
            stream.read_exact(&mut buf)?;
            let id = buf[0];
            let mut mx = [0u8; 8];
            let mut my = [0u8; 8];
            mx.copy_from_slice(&buf[1..9]);
            my.copy_from_slice(&buf[9..17]);
            Msg::MulToHp {
                id,
                mx: u64::from_be_bytes(mx),
                my: u64::from_be_bytes(my),
            }
        }
        MSG_MATMUL_TO_HP => {
            let mut id = [0u8; 1];
            stream.read_exact(&mut id)?;
            let a = read_u64_vec(stream)?;
            let b = read_u64_vec(stream)?;
            Msg::MatMulToHp { id: id[0], a, b }
        }
        MSG_MUL_VEC_TO_HP => {
            let mut id = [0u8; 1];
            stream.read_exact(&mut id)?;
            let mx = read_u64_vec(stream)?;
            let my = read_u64_vec(stream)?;
            Msg::MulVecToHp { id: id[0], mx, my }
        }
        MSG_SHARE => {
            let mut buf = [0u8; 8];
            stream.read_exact(&mut buf)?;
            Msg::Share(u64::from_be_bytes(buf))
        }
        MSG_SHARE_VEC => Msg::ShareVec(read_u64_vec(stream)?),
        MSG_REAL => {
            let mut buf = [0u8; 8];
            stream.read_exact(&mut buf)?;
            Msg::Real(f64::from_bits(u64::from_be_bytes(buf)))
        }
        MSG_REAL_VEC => Msg::RealVec(read_f64_vec(stream)?),
        MSG_REAL_PAIR => {
            let mut buf = [0u8; 16];
            stream.read_exact(&mut buf)?;
            let mut a = [0u8; 8];
            let mut b = [0u8; 8];
            a.copy_from_slice(&buf[..8]);
            b.copy_from_slice(&buf[8..]);
            Msg::RealPair(
                f64::from_bits(u64::from_be_bytes(a)),
                f64::from_bits(u64::from_be_bytes(b)),
            )
        }
        MSG_REAL_PAIR_VEC => {
            let a = read_f64_vec(stream)?;
            let b = read_f64_vec(stream)?;
            Msg::RealPairVec { a, b }
        }
        MSG_BIT => {
            let mut buf = [0u8; 1];
            stream.read_exact(&mut buf)?;
            Msg::Bit(buf[0])
        }
        MSG_BIT_VEC => Msg::BitVec(read_u8_vec(stream)?),
        other => return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown message tag {other}"),
        )),
    };
    let read_nanos = read_start.elapsed().as_nanos() as u64;
    Ok((msg, wait_nanos, read_nanos))
}

pub struct SocketPartyEndpoint {
    stream: std::sync::Mutex<TcpStream>,
}

impl SocketPartyEndpoint {
    pub fn connect(addr: &str, id: u8) -> std::io::Result<Self> {
        let mut stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        stream.write_all(&[id])?;
        Ok(Self {
            stream: std::sync::Mutex::new(stream),
        })
    }
}

impl PartyComm for SocketPartyEndpoint {
    fn send_to_hp(&self, m: Msg) {
        let bytes = msg_wire_len(&m);
        let mut stream = self.stream.lock().expect("socket party mutex poisoned");
        let start = Instant::now();
        write_msg(&mut stream, &m).expect("socket send_to_hp failed");
        record_send(start.elapsed().as_nanos() as u64, bytes);
    }

    fn recv_from_hp(&self) -> Msg {
        let mut stream = self.stream.lock().expect("socket party mutex poisoned");
        let start = Instant::now();
        let (msg, wait_nanos, read_nanos) =
            read_msg(&mut stream).expect("socket recv_from_hp failed");
        let bytes = msg_wire_len(&msg);
        record_recv(start.elapsed().as_nanos() as u64, wait_nanos, read_nanos, bytes);
        msg
    }
}

pub struct SocketHpEndpoint {
    p0: std::sync::Mutex<TcpStream>,
    p1: std::sync::Mutex<TcpStream>,
}

impl SocketHpEndpoint {
    pub fn listen(addr: &str) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let mut p0 = None;
        let mut p1 = None;

        while p0.is_none() || p1.is_none() {
            let (mut stream, _) = listener.accept()?;
            stream.set_nodelay(true)?;
            let mut id = [0u8; 1];
            stream.read_exact(&mut id)?;
            match id[0] {
                0 if p0.is_none() => p0 = Some(stream),
                1 if p1.is_none() => p1 = Some(stream),
                other => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("unexpected or duplicate party id {other}"),
                    ));
                }
            }
        }

        Ok(Self {
            p0: std::sync::Mutex::new(p0.expect("p0 connected")),
            p1: std::sync::Mutex::new(p1.expect("p1 connected")),
        })
    }
}

impl HpComm for SocketHpEndpoint {
    fn recv_from_p0(&self) -> Msg {
        let mut stream = self.p0.lock().expect("socket hp p0 mutex poisoned");
        let start = Instant::now();
        let (msg, wait_nanos, read_nanos) =
            read_msg(&mut stream).expect("socket recv_from_p0 failed");
        let bytes = msg_wire_len(&msg);
        record_recv(start.elapsed().as_nanos() as u64, wait_nanos, read_nanos, bytes);
        msg
    }

    fn recv_from_p1(&self) -> Msg {
        let mut stream = self.p1.lock().expect("socket hp p1 mutex poisoned");
        let start = Instant::now();
        let (msg, wait_nanos, read_nanos) =
            read_msg(&mut stream).expect("socket recv_from_p1 failed");
        let bytes = msg_wire_len(&msg);
        record_recv(start.elapsed().as_nanos() as u64, wait_nanos, read_nanos, bytes);
        msg
    }

    fn recv_from_parties(&self) -> (Msg, Msg) {
        let start = Instant::now();
        std::thread::scope(|scope| {
            let p0 = scope.spawn(|| {
                let mut stream = self.p0.lock().expect("socket hp p0 mutex poisoned");
                let (msg, wait_nanos, read_nanos) =
                    read_msg(&mut stream).expect("socket recv_from_p0 failed");
                let bytes = msg_wire_len(&msg);
                (msg, wait_nanos, read_nanos, bytes)
            });
            let p1 = scope.spawn(|| {
                let mut stream = self.p1.lock().expect("socket hp p1 mutex poisoned");
                let (msg, wait_nanos, read_nanos) =
                    read_msg(&mut stream).expect("socket recv_from_p1 failed");
                let bytes = msg_wire_len(&msg);
                (msg, wait_nanos, read_nanos, bytes)
            });
            let (msg0, wait0, read0, bytes0) =
                p0.join().expect("socket recv_from_p0 thread failed");
            let (msg1, wait1, read1, bytes1) =
                p1.join().expect("socket recv_from_p1 thread failed");
            record_recv_batch(
                2,
                start.elapsed().as_nanos() as u64,
                wait0.max(wait1),
                read0.max(read1),
                bytes0 + bytes1,
            );
            (msg0, msg1)
        })
    }

    fn send_to_p0(&self, m: Msg) {
        let bytes = msg_wire_len(&m);
        let mut stream = self.p0.lock().expect("socket hp p0 mutex poisoned");
        let start = Instant::now();
        write_msg(&mut stream, &m).expect("socket send_to_p0 failed");
        record_send(start.elapsed().as_nanos() as u64, bytes);
    }

    fn send_to_p1(&self, m: Msg) {
        let bytes = msg_wire_len(&m);
        let mut stream = self.p1.lock().expect("socket hp p1 mutex poisoned");
        let start = Instant::now();
        write_msg(&mut stream, &m).expect("socket send_to_p1 failed");
        record_send(start.elapsed().as_nanos() as u64, bytes);
    }

    fn send_to_parties(&self, p0_msg: Msg, p1_msg: Msg) {
        std::thread::scope(|scope| {
            let p0 = scope.spawn(|| {
                let bytes = msg_wire_len(&p0_msg);
                let mut stream = self.p0.lock().expect("socket hp p0 mutex poisoned");
                let start = Instant::now();
                write_msg(&mut stream, &p0_msg).expect("socket send_to_p0 failed");
                record_send(start.elapsed().as_nanos() as u64, bytes);
            });
            let p1 = scope.spawn(|| {
                let bytes = msg_wire_len(&p1_msg);
                let mut stream = self.p1.lock().expect("socket hp p1 mutex poisoned");
                let start = Instant::now();
                write_msg(&mut stream, &p1_msg).expect("socket send_to_p1 failed");
                record_send(start.elapsed().as_nanos() as u64, bytes);
            });
            p0.join().expect("socket send_to_p0 thread failed");
            p1.join().expect("socket send_to_p1 thread failed");
        });
    }
}
