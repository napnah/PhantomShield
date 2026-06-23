//! PyO3 绑定：批处理接口（numpy 数组进出）。
//!
//! 约定：所有协议返回两方份额 `(s0, s1)`，Python 端 `s0 + s1` 重构并验证。
//! 种子用 Python `bytes`（16 字节）传入。

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::simulate;

fn to_seed(b: &[u8]) -> PyResult<[u8; 16]> {
    if b.len() != 16 {
        return Err(PyValueError::new_err(format!(
            "seed 必须为 16 字节，收到 {}",
            b.len()
        )));
    }
    let mut s = [0u8; 16];
    s.copy_from_slice(b);
    Ok(s)
}

/// 并行生成 `count` 个 PRG 原始 64 位字（对齐 Python `PRGSync.next()`）。
#[pyfunction]
pub fn prg_next_batch<'py>(
    py: Python<'py>,
    seed: Vec<u8>,
    count: usize,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let seed = to_seed(&seed)?;
    let out = py.allow_threads(|| simulate::prg_next_batch(&seed, count));
    Ok(out.into_pyarray_bound(py))
}

/// 批量整数环安全乘法。
#[pyfunction]
pub fn multiply<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray1<u64>,
    x1: PyReadonlyArray1<u64>,
    y0: PyReadonlyArray1<u64>,
    y1: PyReadonlyArray1<u64>,
    seed_shared: Vec<u8>,
    seed_hp: Vec<u8>,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let seed_shared = to_seed(&seed_shared)?;
    let seed_hp = to_seed(&seed_hp)?;
    let x0 = x0.as_slice()?;
    let x1 = x1.as_slice()?;
    let y0 = y0.as_slice()?;
    let y1 = y1.as_slice()?;
    let (s0, s1) =
        py.allow_threads(|| simulate::multiply_batch(x0, x1, y0, y1, &seed_shared, &seed_hp));
    Ok((s0.into_pyarray_bound(py), s1.into_pyarray_bound(py)))
}

/// 批量安全指数 e^x。
#[pyfunction]
pub fn exp<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray1<f64>,
    x1: PyReadonlyArray1<f64>,
    seed_shared: Vec<u8>,
    seed_hp: Vec<u8>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let seed_shared = to_seed(&seed_shared)?;
    let seed_hp = to_seed(&seed_hp)?;
    let x0 = x0.as_slice()?;
    let x1 = x1.as_slice()?;
    let (e0, e1) = py.allow_threads(|| simulate::exp_batch(x0, x1, &seed_shared, &seed_hp));
    Ok((e0.into_pyarray_bound(py), e1.into_pyarray_bound(py)))
}

/// 批量安全 Sigmoid。
#[pyfunction]
pub fn sigmoid<'py>(
    py: Python<'py>,
    z0: PyReadonlyArray1<f64>,
    z1: PyReadonlyArray1<f64>,
    seed_shared: Vec<u8>,
    seed_hp: Vec<u8>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let seed_shared = to_seed(&seed_shared)?;
    let seed_hp = to_seed(&seed_hp)?;
    let z0 = z0.as_slice()?;
    let z1 = z1.as_slice()?;
    let (s0, s1) = py.allow_threads(|| simulate::sigmoid_batch(z0, z1, &seed_shared, &seed_hp));
    Ok((s0.into_pyarray_bound(py), s1.into_pyarray_bound(py)))
}

/// 批量安全 GeLU。
#[pyfunction]
pub fn gelu<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray1<f64>,
    x1: PyReadonlyArray1<f64>,
    seed_shared: Vec<u8>,
    seed_hp: Vec<u8>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let seed_shared = to_seed(&seed_shared)?;
    let seed_hp = to_seed(&seed_hp)?;
    let x0 = x0.as_slice()?;
    let x1 = x1.as_slice()?;
    let (g0, g1) = py.allow_threads(|| simulate::gelu_batch(x0, x1, &seed_shared, &seed_hp));
    Ok((g0.into_pyarray_bound(py), g1.into_pyarray_bound(py)))
}

/// 批量安全 Softmax（按行）。输入展平的 `n×k` 行主序份额，返回展平两方份额。
#[pyfunction]
pub fn softmax<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray1<f64>,
    x1: PyReadonlyArray1<f64>,
    n: usize,
    k: usize,
    seed_shared: Vec<u8>,
    seed_hp: Vec<u8>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let seed_shared = to_seed(&seed_shared)?;
    let seed_hp = to_seed(&seed_hp)?;
    let x0 = x0.as_slice()?;
    let x1 = x1.as_slice()?;
    if x0.len() != n * k || x1.len() != n * k {
        return Err(PyValueError::new_err("输入长度必须等于 n*k"));
    }
    let (s0, s1) =
        py.allow_threads(|| simulate::softmax_batch(x0, x1, n, k, &seed_shared, &seed_hp));
    Ok((s0.into_pyarray_bound(py), s1.into_pyarray_bound(py)))
}
