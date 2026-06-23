//! 通信抽象：`Comm` trait + 进程内 mock（std::sync::mpsc + 线程）。
//!
//! 结构对齐 Python 的三方消息流（每个发送方→HP 用独立信箱，HP→各方各一条），
//! 便于未来替换为真实 socket 实现。PyO3 高速路径不走该层（见 `simulate.rs`），
//! 此处用于忠实复现协议轮次与做结构性验证。

use std::sync::mpsc::{channel, Receiver, Sender};

/// 三方之间传递的消息（覆盖当前需通过 Comm 层演示的协议）。
#[derive(Debug, Clone)]
pub enum Msg {
    /// 乘法协议：P_i → HP 的掩码值。
    MulToHp { id: u8, mx: u64, my: u64 },
    /// HP → P_i 的整数环份额。
    Share(u64),
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
    fn send_to_p0(&self, m: Msg);
    fn send_to_p1(&self, m: Msg);
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
