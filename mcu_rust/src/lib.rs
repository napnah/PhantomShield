//! mcu_rust：Rust 高效实现的 mcu_core 协议核心 + PyO3 绑定。
//!
//! 层次：
//!   - `prg`        AES-CTR 同步 PRG（对齐 Python，字节级）
//!   - `ring`       Z_2^64 环运算
//!   - `channel`    Comm 抽象 + 进程内 mock（未来接 socket）
//!   - `protocols`  6 个协议的纯数学核心（+ 乘法的 Comm 路径）
//!   - `simulate`   融合批处理 + rayon 并行（PyO3 高速路径）
//!   - `bindings`   PyO3 批处理接口

pub mod bindings;
pub mod channel;
pub mod prg;
pub mod protocols;
pub mod real_protocols;
pub mod ring;
pub mod simulate;
pub mod tensor;

use pyo3::prelude::*;

/// Python 扩展模块 `mcu_rust`。
#[pymodule]
fn mcu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust 高效实现的 mcu_core 协议核心（PyO3 绑定）")?;
    m.add_function(wrap_pyfunction!(bindings::prg_next_batch, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::multiply, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::exp, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::gelu, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::softmax, m)?)?;
    Ok(())
}
