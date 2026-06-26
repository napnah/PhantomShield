from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch


SEQ = 4
HIDDEN = 16
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
FFN = 32


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def run_op(op: str):
    import crypten

    torch.manual_seed(2026)
    if op == "elemul":
        length = env_int("CRYPTEN_LEN", 64)
        x = crypten.cryptensor(torch.randn(length))
        y = crypten.cryptensor(torch.randn(length))
        return (x * y).get_plain_text()
    if op == "matmul":
        m = env_int("CRYPTEN_M", 4)
        k = env_int("CRYPTEN_K", 16)
        n = env_int("CRYPTEN_N", 16)
        a = crypten.cryptensor(torch.randn(m, k))
        b = crypten.cryptensor(torch.randn(k, n))
        return a.matmul(b).get_plain_text()
    if op == "exp":
        length = env_int("CRYPTEN_LEN", 64)
        return crypten.cryptensor(torch.randn(length)).exp().get_plain_text()
    if op == "sigmoid":
        length = env_int("CRYPTEN_LEN", 64)
        return crypten.cryptensor(torch.randn(length)).sigmoid().get_plain_text()
    if op == "gelu":
        length = env_int("CRYPTEN_LEN", 64)
        x = crypten.cryptensor(torch.randn(length))
        return (x * (x * 1.702).sigmoid()).get_plain_text()
    if op == "softmax":
        rows = env_int("CRYPTEN_SOFTMAX_ROWS", 16)
        cols = env_int("CRYPTEN_SOFTMAX_COLS", 4)
        return crypten.cryptensor(torch.randn(rows, cols)).softmax(dim=-1).get_plain_text()
    if op == "attention":
        batch = env_int("CRYPTEN_BATCH", 1)
        x = crypten.cryptensor(torch.randn(batch, SEQ, HIDDEN))
        wq = crypten.cryptensor(torch.randn(HIDDEN, HIDDEN))
        wk = crypten.cryptensor(torch.randn(HIDDEN, HIDDEN))
        wv = crypten.cryptensor(torch.randn(HIDDEN, HIDDEN))
        wo = crypten.cryptensor(torch.randn(HIDDEN, HIDDEN))
        scale = HEAD_DIM**0.5
        q = x.matmul(wq).view(batch, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        k = x.matmul(wk).view(batch, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        v = x.matmul(wv).view(batch, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        scores = q.matmul(k.transpose(2, 3)) / scale
        probs = scores.softmax(dim=-1)
        ctx = probs.matmul(v).transpose(1, 2).reshape(batch, SEQ, HIDDEN)
        return ctx.matmul(wo).get_plain_text()
    if op == "ffn":
        batch = env_int("CRYPTEN_BATCH", 1)
        x = crypten.cryptensor(torch.randn(batch, SEQ, HIDDEN))
        w1 = crypten.cryptensor(torch.randn(HIDDEN, FFN))
        b1 = crypten.cryptensor(torch.randn(FFN))
        w2 = crypten.cryptensor(torch.randn(FFN, HIDDEN))
        b2 = crypten.cryptensor(torch.randn(HIDDEN))
        h = x.matmul(w1) + b1
        h = h * (h * 1.702).sigmoid()
        return (h.matmul(w2) + b2).get_plain_text()
    raise ValueError(op)


def run_plain_op(op: str):
    torch.manual_seed(2026)
    if op == "elemul":
        length = env_int("CRYPTEN_LEN", 64)
        x = torch.randn(length)
        y = torch.randn(length)
        return x * y
    if op == "matmul":
        m = env_int("CRYPTEN_M", 4)
        k = env_int("CRYPTEN_K", 16)
        n = env_int("CRYPTEN_N", 16)
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        return a.matmul(b)
    if op == "exp":
        length = env_int("CRYPTEN_LEN", 64)
        return torch.randn(length).exp()
    if op == "sigmoid":
        length = env_int("CRYPTEN_LEN", 64)
        return torch.randn(length).sigmoid()
    if op == "gelu":
        length = env_int("CRYPTEN_LEN", 64)
        x = torch.randn(length)
        return x * (x * 1.702).sigmoid()
    if op == "softmax":
        rows = env_int("CRYPTEN_SOFTMAX_ROWS", 16)
        cols = env_int("CRYPTEN_SOFTMAX_COLS", 4)
        return torch.randn(rows, cols).softmax(dim=-1)
    if op == "attention":
        batch = env_int("CRYPTEN_BATCH", 1)
        x = torch.randn(batch, SEQ, HIDDEN)
        wq = torch.randn(HIDDEN, HIDDEN)
        wk = torch.randn(HIDDEN, HIDDEN)
        wv = torch.randn(HIDDEN, HIDDEN)
        wo = torch.randn(HIDDEN, HIDDEN)
        scale = HEAD_DIM**0.5
        q = x.matmul(wq).view(batch, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        k = x.matmul(wk).view(batch, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        v = x.matmul(wv).view(batch, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        scores = q.matmul(k.transpose(2, 3)) / scale
        probs = scores.softmax(dim=-1)
        ctx = probs.matmul(v).transpose(1, 2).reshape(batch, SEQ, HIDDEN)
        return ctx.matmul(wo)
    if op == "ffn":
        batch = env_int("CRYPTEN_BATCH", 1)
        x = torch.randn(batch, SEQ, HIDDEN)
        w1 = torch.randn(HIDDEN, FFN)
        b1 = torch.randn(FFN)
        w2 = torch.randn(FFN, HIDDEN)
        b2 = torch.randn(HIDDEN)
        h = x.matmul(w1) + b1
        h = h * (h * 1.702).sigmoid()
        return h.matmul(w2) + b2
    raise ValueError(op)


def time_plain(op: str, repeat: int) -> tuple[list[float], list[float]]:
    run_plain_op(op)
    times = []
    sums = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = run_plain_op(op)
        times.append(time.perf_counter() - t0)
        sums.append(float(out.sum()))
    return times, sums


def main() -> int:
    if os.environ.get("CRYPTEN_START_DELAY"):
        time.sleep(float(os.environ["CRYPTEN_START_DELAY"]))

    import crypten

    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 2)
    op = os.environ.get("CRYPTEN_OP", "elemul")
    repeat = env_int("CRYPTEN_REPEAT", 5)
    out_dir = Path(os.environ.get("CRYPTEN_OUT_DIR", "/workspace/out/default"))
    out_dir.mkdir(parents=True, exist_ok=True)

    crypten.init()
    plain_times = []
    plain_sums = []
    if rank == 0:
        plain_times, plain_sums = time_plain(op, repeat)
    run_op(op)
    times = []
    sums = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = run_op(op)
        times.append(time.perf_counter() - t0)
        sums.append(float(out.sum()))
    crypten.uninit()

    payload = {
        "rank": rank,
        "world_size": world_size,
        "op": op,
        "repeat": repeat,
        "times_s": times,
        "median_s": statistics.median(times),
        "plain_times_s": plain_times,
        "plain_median_s": statistics.median(plain_times) if plain_times else None,
        "sum_last": sums[-1] if sums else None,
        "plain_sum_last": plain_sums[-1] if plain_sums else None,
        "rendezvous": os.environ.get("RENDEZVOUS", ""),
        "backend": os.environ.get("DISTRIBUTED_BACKEND", ""),
    }
    (out_dir / f"rank{rank}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[crypten-rank{rank}] done: {payload['median_s']:.9f}s op={op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
