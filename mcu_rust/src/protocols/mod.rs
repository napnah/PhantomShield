//! MCU 协议核心。
//!
//! 每个协议提供「纯数学核心」（无 IO，供融合批处理与 Comm 路径共用），
//! 数值参数严格对齐 Python `mcu_core`：
//!   - 超越函数实数模 `MOD = 256.0`；
//!   - wrap ∈ {-1,0,1}（|x| < MOD 假设下）；
//!   - HP 把 E 拆成两个正份额 `s0 = u*E, s1 = (1-u)*E` 以避免灾难性相消。

pub mod multiply;
pub mod wrap;
pub mod exp;
pub mod sigmoid;
pub mod softmax;
pub mod gelu;

/// 超越函数协议统一实数模（对齐 Python `MOD`）。
pub const MOD: f64 = 256.0;

/// GeLU 的 sigmoid 近似系数（对齐 Python `GELU_COEF`）。
pub const GELU_COEF: f64 = 1.702;
