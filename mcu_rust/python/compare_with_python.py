"""
Rust mcu_rust 与 Python mcu_core 交叉验证 + 加速比基准。

运行：
    python mcu_rust/python/compare_with_python.py

内容：
  1) PRG 字节级对齐：mcu_rust.prg_next_batch vs Python PRGSync.next（逐位相等）
  2) 正确性：各协议重构值 (s0+s1) 与明文真值对比（容差）
  3) 加速比：Rust 批处理 vs Python mcu_core 标量协议（同等元素数）
"""
import os
import sys
import time

import numpy as np

# 让 Python 能找到 mcu_core 包
_HERE = os.path.dirname(os.path.abspath(__file__))
_MCU_CORE_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "mcu_core"))
if _MCU_CORE_ROOT not in sys.path:
    sys.path.insert(0, _MCU_CORE_ROOT)

import mcu_rust  # Rust 扩展
from mcu_core.prg_sync import PRGSync
from mcu_core.mock_comm import make_mock_comm
from mcu_core.protocols.exponential import ExpParty, ExpHP, _run_three_party
from mcu_core.protocols.gelu import GeluParty, GeluHP, GELU_COEF
from mcu_core.protocols.softmax import SoftmaxParty, SoftmaxHP

SEED_SHARED = bytes(range(16))
SEED_HP = bytes(range(16, 32))

OK = "[OK]"
FAIL = "[FAIL]"


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# --------------------------------------------------------------------------- #
# 1) PRG 字节级对齐
# --------------------------------------------------------------------------- #
def check_prg(n=2000):
    section("1) PRG 字节级对齐 (Rust vs Python)")
    rust_vals = mcu_rust.prg_next_batch(SEED_SHARED, n)
    p = PRGSync(SEED_SHARED)
    py_vals = np.fromiter((p.next() for _ in range(n)), dtype=np.uint64, count=n)
    exact = bool(np.array_equal(rust_vals.astype(np.uint64), py_vals))
    print(f"  样本数 = {n}")
    print(f"  前 3 个 Rust : {list(rust_vals[:3])}")
    print(f"  前 3 个 Py   : {list(py_vals[:3])}")
    print(f"  逐位相等: {OK if exact else FAIL}")
    return exact


# --------------------------------------------------------------------------- #
# 2) 正确性（重构值 vs 明文）
# --------------------------------------------------------------------------- #
def split_shares(x, lo=-50.0, hi=50.0, rng=None):
    rng = rng or np.random.default_rng(0)
    x0 = rng.uniform(lo, hi, size=x.shape)
    x1 = x - x0
    return x0, x1


def check_correctness():
    section("2) 正确性：Rust 重构值 vs 明文真值")
    rng = np.random.default_rng(42)
    results = []

    # exp
    x = rng.uniform(-10, 10, size=20000)
    x0, x1 = split_shares(x, -100, 100, rng)
    e0, e1 = mcu_rust.exp(x0, x1, SEED_SHARED, SEED_HP)
    err = np.abs((e0 + e1) - np.exp(x))
    ok = err.mean() < 1e-4
    print(f"  exp     : 平均误差={err.mean():.3e} 最大={err.max():.3e}  "
          f"{OK if ok else FAIL} (<1e-4 平均)")
    results.append(ok)

    # sigmoid
    z = rng.uniform(-8, 8, size=20000)
    z0, z1 = split_shares(z, -50, 50, rng)
    s0, s1 = mcu_rust.sigmoid(z0, z1, SEED_SHARED, SEED_HP)
    truth = 1.0 / (1.0 + np.exp(-z))
    err = np.abs((s0 + s1) - truth)
    ok = err.max() < 1e-4
    print(f"  sigmoid : 平均误差={err.mean():.3e} 最大={err.max():.3e}  "
          f"{OK if ok else FAIL} (<1e-4 最大)")
    results.append(ok)

    # gelu
    x = rng.uniform(-5, 5, size=20000)
    x0, x1 = split_shares(x, -50, 50, rng)
    g0, g1 = mcu_rust.gelu(x0, x1, SEED_SHARED, SEED_HP)
    truth = x * (1.0 / (1.0 + np.exp(-(GELU_COEF * x))))
    err = np.abs((g0 + g1) - truth)
    ok = err.mean() < 1e-3
    print(f"  gelu    : 平均误差={err.mean():.3e} 最大={err.max():.3e}  "
          f"{OK if ok else FAIL} (<1e-3 平均)")
    results.append(ok)

    # softmax
    n, k = 2000, 16
    x = rng.uniform(-8, 8, size=(n, k))
    x0 = rng.uniform(-50, 50, size=(n, k))
    x1 = x - x0
    sm0, sm1 = mcu_rust.softmax(
        x0.ravel(), x1.ravel(), n, k, SEED_SHARED, SEED_HP
    )
    recon = (sm0 + sm1).reshape(n, k)
    ex = np.exp(x - x.max(axis=1, keepdims=True))
    truth = ex / ex.sum(axis=1, keepdims=True)
    err = np.abs(recon - truth)
    ok = err.max() < 1e-4
    print(f"  softmax : 平均误差={err.mean():.3e} 最大={err.max():.3e}  "
          f"{OK if ok else FAIL} (<1e-4 最大)")
    results.append(ok)

    # multiply（整数环，精确）
    n = 20000
    xi = rng.integers(0, 10**9, size=n, dtype=np.uint64)
    yi = rng.integers(0, 10**9, size=n, dtype=np.uint64)
    xi0 = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    yi0 = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    xi1 = xi - xi0  # uint64 回绕
    yi1 = yi - yi0
    m0, m1 = mcu_rust.multiply(xi0, xi1, yi0, yi1, SEED_SHARED, SEED_HP)
    recon = (m0 + m1)  # uint64 回绕
    truth = xi * yi    # uint64 回绕
    exact = bool(np.array_equal(recon, truth))
    print(f"  multiply: 逐位精确={exact}  {OK if exact else FAIL}")
    results.append(exact)

    return all(results)


# --------------------------------------------------------------------------- #
# 2b) 与 Python mcu_core 协议（非明文）逐元素对齐抽查
# --------------------------------------------------------------------------- #
def check_vs_python_protocol(n=200):
    section("2b) Rust vs Python mcu_core 协议（重构值一致，抽查 %d 个）" % n)
    rng = np.random.default_rng(7)
    xs = rng.uniform(-10, 10, size=n)
    sp = rng.uniform(-100, 100, size=n)
    x0 = sp
    x1 = xs - sp

    # Rust 批处理
    re0, re1 = mcu_rust.exp(x0, x1, SEED_SHARED, SEED_HP)
    rust_recon = re0 + re1

    # Python mcu_core 协议（逐元素，threaded mock）
    py_recon = np.empty(n)
    for i in range(n):
        prg0_p0 = PRGSync(SEED_SHARED)
        prg0_p1 = PRGSync(SEED_SHARED)
        asprg_p0 = PRGSync(SEED_HP)
        c0, c1, chp = make_mock_comm()
        ep0 = ExpParty(0, prg0_p0, c0)
        ep1 = ExpParty(1, prg0_p1, c1)
        ehp = ExpHP(asprg_p0, chp)
        out = {}
        _run_three_party(
            lambda: out.__setitem__("e0", ep0.exp(x0[i])),
            lambda: out.__setitem__("e1", ep1.exp(x1[i])),
            lambda: ehp.serve_exp(),
        )
        py_recon[i] = out["e0"] + out["e1"]

    diff = np.abs(rust_recon - py_recon)
    ok = diff.max() < 1e-6
    print(f"  exp 重构值 |Rust - Python| 最大={diff.max():.3e}  "
          f"{OK if ok else FAIL} (<1e-6)")
    return ok


# --------------------------------------------------------------------------- #
# 3) 加速比基准
# --------------------------------------------------------------------------- #
def _py_exp_loop(x0, x1):
    n = len(x0)
    for i in range(n):
        prg0_p0 = PRGSync(SEED_SHARED)
        prg0_p1 = PRGSync(SEED_SHARED)
        asprg_p0 = PRGSync(SEED_HP)
        c0, c1, chp = make_mock_comm()
        ep0 = ExpParty(0, prg0_p0, c0)
        ep1 = ExpParty(1, prg0_p1, c1)
        ehp = ExpHP(asprg_p0, chp)
        out = {}
        _run_three_party(
            lambda: out.__setitem__("e0", ep0.exp(x0[i])),
            lambda: out.__setitem__("e1", ep1.exp(x1[i])),
            lambda: ehp.serve_exp(),
        )


def _py_gelu_loop(x0, x1):
    n = len(x0)
    for i in range(n):
        prg0_p0 = PRGSync(SEED_SHARED)
        prg0_p1 = PRGSync(SEED_SHARED)
        asprg_p0 = PRGSync(SEED_HP)
        c0, c1, chp = make_mock_comm()
        gp0 = GeluParty(0, prg0_p0, c0)
        gp1 = GeluParty(1, prg0_p1, c1)
        ghp = GeluHP(asprg_p0, chp)
        out = {}
        _run_three_party(
            lambda: out.__setitem__("g0", gp0.gelu(x0[i])),
            lambda: out.__setitem__("g1", gp1.gelu(x1[i])),
            lambda: ghp.serve_gelu(),
        )


def bench():
    section("3) 加速比：Rust 批处理 vs Python mcu_core 标量协议")
    rng = np.random.default_rng(123)

    # ---- exp ----
    n_py = 1000               # Python 协议较慢，用较小规模计时
    x = rng.uniform(-10, 10, size=n_py)
    x0 = rng.uniform(-100, 100, size=n_py)
    x1 = x - x0

    t0 = time.perf_counter()
    _py_exp_loop(x0, x1)
    t_py = time.perf_counter() - t0

    # Rust 同规模
    t0 = time.perf_counter()
    for _ in range(20):       # 多跑几次取均值（Rust 太快）
        mcu_rust.exp(x0, x1, SEED_SHARED, SEED_HP)
    t_rust_same = (time.perf_counter() - t0) / 20

    print(f"  [exp]  N={n_py}")
    print(f"    Python mcu_core : {t_py*1e3:8.1f} ms  "
          f"({t_py/n_py*1e6:.1f} us/elem)")
    print(f"    Rust 批处理      : {t_rust_same*1e3:8.3f} ms  "
          f"({t_rust_same/n_py*1e6:.3f} us/elem)")
    print(f"    >>> 加速比 ≈ {t_py/max(t_rust_same,1e-12):,.0f}x")

    # ---- gelu ----
    n_py = 500
    x = rng.uniform(-5, 5, size=n_py)
    x0 = rng.uniform(-50, 50, size=n_py)
    x1 = x - x0

    t0 = time.perf_counter()
    _py_gelu_loop(x0, x1)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        mcu_rust.gelu(x0, x1, SEED_SHARED, SEED_HP)
    t_rust_same = (time.perf_counter() - t0) / 20

    print(f"  [gelu] N={n_py}")
    print(f"    Python mcu_core : {t_py*1e3:8.1f} ms  "
          f"({t_py/n_py*1e6:.1f} us/elem)")
    print(f"    Rust 批处理      : {t_rust_same*1e3:8.3f} ms  "
          f"({t_rust_same/n_py*1e6:.3f} us/elem)")
    print(f"    >>> 加速比 ≈ {t_py/max(t_rust_same,1e-12):,.0f}x")

    # ---- Rust 大规模吞吐（展示批处理能力）----
    big = 512 * 512
    x = rng.uniform(-10, 10, size=big)
    x0 = rng.uniform(-100, 100, size=big)
    x1 = x - x0
    t0 = time.perf_counter()
    mcu_rust.exp(x0, x1, SEED_SHARED, SEED_HP)
    t_big = time.perf_counter() - t0
    print(f"  [exp]  Rust 大规模 N={big:,} : {t_big*1e3:.1f} ms  "
          f"({big/t_big/1e6:.1f} M elem/s)")


def main():
    a = check_prg()
    b = check_correctness()
    c = check_vs_python_protocol()
    bench()

    section("总结")
    print(f"  PRG 字节对齐 : {OK if a else FAIL}")
    print(f"  正确性       : {OK if b else FAIL}")
    print(f"  协议交叉验证 : {OK if c else FAIL}")
    all_ok = a and b and c
    print(f"\n  全部通过: {OK if all_ok else FAIL}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
