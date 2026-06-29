from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


SEQ = 4
HIDDEN = 16
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
FFN = 32
LABELS = ["negative", "positive"]


DEFAULT_BERT_SAMPLES = [
    {"text": "This movie is wonderful and heartwarming.", "label": 1},
    {"text": "A complete waste of time, absolutely terrible.", "label": 0},
]


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def load_bert_samples() -> list[dict]:
    path = os.environ.get("CRYPTEN_INPUT_JSON")
    if not path:
        return DEFAULT_BERT_SAMPLES
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("samples", [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"no BERT samples found in {path}")
    out = []
    for item in data:
        if isinstance(item, str):
            out.append({"text": item, "label": None})
        else:
            out.append({"text": str(item["text"]), "label": item.get("label")})
    return out


def crypten_security_profile(op: str) -> dict:
    if op != "bert_full":
        return {"mode": "crypten_2pc", "notes": ["Synthetic operator benchmark."]}
    nonlinear = os.environ.get("CRYPTEN_BERT_NONLINEAR", "native").strip().lower()
    return {
        "mode": "crypten_docker_2pc_bert_full_engineering_baseline",
        "nonlinear": nonlinear,
        "secure_components": [
            "linear projections, attention score/value matmul, FFN matmul, classifier matmul through CrypTen tensors",
            "attention softmax and classifier softmax through CrypTen tensors when CRYPTEN_BERT_NONLINEAR=native",
            "sigmoid-GELU approximation through CrypTen tensors when CRYPTEN_BERT_NONLINEAR=native",
        ],
        "plaintext_components": [
            "tokenization and embedding lookup",
            "legacy_two_quad mode reconstructs scores before two_quad",
            "LayerNorm variance inverse currently reconstructs variance",
            "attention context and pooler output are reconstructed before later steps",
            "both Docker ranks load the same local model checkpoint in this baseline",
        ],
        "status": "full BERT Docker execution path, not a final threat-model-complete secure inference.",
    }


def load_bert_model_and_tokenizer():
    from transformers import BertForSequenceClassification, BertTokenizer

    root = Path("/workspace")
    ckpt = root / "checkpoints" / "bert-sst2"
    model_dir = root / "bert-base-uncased"
    path = ckpt if (ckpt / "config.json").is_file() else model_dir
    tokenizer = BertTokenizer.from_pretrained(str(path))
    model = BertForSequenceClassification.from_pretrained(str(path), num_labels=2).eval()
    return model, tokenizer


def run_bert_full_once(model, tokenizer, samples: list[dict]) -> dict:
    from transformer.mcu_bert_crypten_full import classify_crypten_full
    from transformer.plaintext_bert import classify_plaintext_hf

    max_seq_len = env_int("CRYPTEN_MAX_SEQ_LEN", 16)
    max_layers = env_int("CRYPTEN_MAX_LAYERS", 12)
    predictions = []
    sample_times = []
    plain_times = []
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            text = sample["text"]
            t0 = time.perf_counter()
            label, probs, conf = classify_crypten_full(
                model, tokenizer, text, "cpu", max_seq_len=max_seq_len, max_layers=max_layers
            )
            elapsed = time.perf_counter() - t0
            sample_times.append(elapsed)

            pt0 = time.perf_counter()
            plain_label, plain_probs = classify_plaintext_hf(model, tokenizer, text, "cpu", max_seq_len)
            plain_elapsed = time.perf_counter() - pt0
            plain_times.append(plain_elapsed)

            pred = LABELS.index(label)
            plain_pred = LABELS.index(plain_label)
            predictions.append(
                {
                    "sample_id": idx,
                    "text": text,
                    "gold_label": sample.get("label"),
                    "prediction": pred,
                    "label": label,
                    "probabilities": [float(x) for x in probs],
                    "confidence": float(conf),
                    "plain_prediction": plain_pred,
                    "plain_label": plain_label,
                    "plain_probabilities": [float(x) for x in plain_probs],
                    "latency_s": elapsed,
                    "plain_latency_s": plain_elapsed,
                }
            )
    return {
        "sample_times_s": sample_times,
        "plain_sample_times_s": plain_times,
        "predictions": predictions,
        "sum_last": float(sum(sum(p["probabilities"]) for p in predictions)),
    }


def run_op(op: str):
    import crypten

    torch.manual_seed(2026)
    if op == "bert_full":
        model, tokenizer = load_bert_model_and_tokenizer()
        return run_bert_full_once(model, tokenizer, load_bert_samples())
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
    if op == "bert_full":
        model, tokenizer = load_bert_model_and_tokenizer()
        samples = load_bert_samples()
        from transformer.plaintext_bert import classify_plaintext_hf

        out = []
        for sample in samples:
            label, probs = classify_plaintext_hf(
                model, tokenizer, sample["text"], "cpu", env_int("CRYPTEN_MAX_SEQ_LEN", 16)
            )
            out.append(sum(probs) + LABELS.index(label))
        return torch.tensor(out)
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
    if rank == 0 and op != "bert_full":
        plain_times, plain_sums = time_plain(op, repeat)
    run_op(op)
    times = []
    sums = []
    bert_runs = []
    bert_plain_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = run_op(op)
        times.append(time.perf_counter() - t0)
        if op == "bert_full":
            bert_runs.append(out)
            sums.append(float(out["sum_last"]))
            bert_plain_times.extend(out["plain_sample_times_s"])
        else:
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
        "security_profile": crypten_security_profile(op),
    }
    if op == "bert_full":
        last = bert_runs[-1] if bert_runs else {}
        payload.update(
            {
                "max_seq_len": env_int("CRYPTEN_MAX_SEQ_LEN", 16),
                "max_layers": env_int("CRYPTEN_MAX_LAYERS", 12),
                "n_samples": len(last.get("predictions", [])),
                "predictions": last.get("predictions", []),
                "sample_times_s": last.get("sample_times_s", []),
                "plain_sample_times_s": last.get("plain_sample_times_s", []),
                "plain_median_s": statistics.median(bert_plain_times) if bert_plain_times else None,
            }
        )
    (out_dir / f"rank{rank}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[crypten-rank{rank}] done: {payload['median_s']:.9f}s op={op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
