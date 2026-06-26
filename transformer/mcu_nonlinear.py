"""
份额拆分 / 重建、2Quad 近似、mcu_rust 非线性封装。
"""
from __future__ import annotations

import numpy as np
import torch

SEED_SHARED = bytes(range(16))
SEED_HP = bytes(range(16, 32))

try:
    import mcu_rust as _mcu_rust
    HAS_MCU_RUST = True
except ImportError:
    _mcu_rust = None
    HAS_MCU_RUST = False


def two_quad(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """SecFormer 2Quad：替换 Softmax 的二次归一化。"""
    num = (x + c) ** 2
    den = num.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return num / den


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """CrypTen / SecFormer 使用的 sigmoid GeLU 近似。"""
    return x * torch.sigmoid(x * 1.702)


def split_shares_f64(x: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(42)
    x0 = rng.uniform(-1.0, 1.0, size=x.shape).astype(np.float64)
    x1 = (x.astype(np.float64) - x0)
    return x0, x1


def reconstruct_f64(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    return s0 + s1


def mcu_rust_softmax_rows(
    scores: torch.Tensor,
    seed_shared: bytes = SEED_SHARED,
    seed_hp: bytes = SEED_HP,
) -> torch.Tensor:
    """
    scores: (..., k) 在最后一维做 softmax。
    返回重建后的概率（明文，用于推理链）。
    """
    if not HAS_MCU_RUST:
        raise ImportError("mcu_rust 未安装，请在 mcu_rust/ 下 maturin develop --release")

    orig_shape = scores.shape
    device = scores.device
    k = orig_shape[-1]
    flat = scores.detach().cpu().numpy().astype(np.float64).reshape(-1, k)
    n = flat.shape[0]
    x0, x1 = split_shares_f64(flat)
    s0, s1 = _mcu_rust.softmax(
        x0.ravel(), x1.ravel(), n, k, seed_shared, seed_hp
    )
    out = reconstruct_f64(s0, s1).reshape(n, k)
    return torch.from_numpy(out).reshape(orig_shape).to(device=device, dtype=scores.dtype)


def mcu_rust_gelu(
    x: torch.Tensor,
    seed_shared: bytes = SEED_SHARED,
    seed_hp: bytes = SEED_HP,
) -> torch.Tensor:
    if not HAS_MCU_RUST:
        raise ImportError("mcu_rust 未安装")

    device = x.device
    flat = x.detach().cpu().numpy().astype(np.float64).ravel()
    x0, x1 = split_shares_f64(flat.reshape(-1, 1))
    g0, g1 = _mcu_rust.gelu(x0.ravel(), x1.ravel(), seed_shared, seed_hp)
    out = reconstruct_f64(g0, g1)
    return torch.from_numpy(out).reshape(x.shape).to(device=device, dtype=x.dtype)


def verify_mcu_rust_prg(n: int = 100) -> bool:
    """供 /api/rust/verify 使用。"""
    if not HAS_MCU_RUST:
        return False
    import sys
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mcu_core = os.path.join(root, "mcu_core")
    if mcu_core not in sys.path:
        sys.path.insert(0, mcu_core)
    from mcu_core.prg_sync import PRGSync

    rust_vals = _mcu_rust.prg_next_batch(SEED_SHARED, n)
    p = PRGSync(SEED_SHARED)
    py_vals = np.fromiter((p.next() for _ in range(n)), dtype=np.uint64, count=n)
    return bool(np.array_equal(rust_vals.astype(np.uint64), py_vals))
