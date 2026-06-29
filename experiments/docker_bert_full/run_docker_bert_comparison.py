from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker" / "docker-compose.mpc.yml"
LABELS = ["negative", "positive"]
EPS = 1e-12

DEFAULT_SAMPLES = [
    {"text": "This movie is wonderful and heartwarming.", "label": 1},
    {"text": "A complete waste of time, absolutely terrible.", "label": 0},
    {"text": "I loved every minute of it!", "label": 1},
    {"text": "Terrible and boring film.", "label": 0},
    {"text": "Brilliant acting and a great story.", "label": 1},
    {"text": "A delightful film with amazing performances.", "label": 1},
    {"text": "Boring, predictable, and poorly acted.", "label": 0},
    {"text": "I hated this film from start to finish.", "label": 0},
    {"text": "Painfully slow and utterly pointless.", "label": 0},
    {"text": "Nothing made sense and it was dull.", "label": 0},
]


def run(cmd: list[str], env: dict[str, str], timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def compose_base(project: str) -> list[str]:
    return ["docker", "compose", "-p", project, "-f", str(COMPOSE)]


def cleanup(project: str, env: dict[str, str]) -> None:
    run([*compose_base(project), "down", "--remove-orphans"], env, timeout=120, check=False)


def docker_bind_root(out_host: Path, timestamp: str) -> Path:
    override = os.environ.get("PHANTOM_DOCKER_BIND_HOST")
    if override:
        path = Path(override)
    elif os.name == "nt":
        path = Path(tempfile.gettempdir()) / "phantomshield_docker_bert_out" / timestamp
    else:
        path = out_host
    path.mkdir(parents=True, exist_ok=True)
    return path


def docker_model_roots(stage_models: bool) -> tuple[Path | None, Path | None]:
    override_base = os.environ.get("PHANTOM_BERT_BASE_HOST")
    override_ckpt = os.environ.get("PHANTOM_CHECKPOINTS_HOST")
    if override_base and override_ckpt:
        return Path(override_base), Path(override_ckpt)
    if not stage_models:
        return None, None
    if os.name != "nt":
        return ROOT / "bert-base-uncased", ROOT / "checkpoints"

    cache_root = Path(os.environ.get("PHANTOM_DOCKER_MODEL_CACHE", Path(tempfile.gettempdir()) / "phantomshield_docker_bert_models"))
    staged = cache_root / "shared"
    base_dst = staged / "bert-base-uncased"
    ckpt_dst = staged / "checkpoints"
    if not base_dst.exists():
        shutil.copytree(ROOT / "bert-base-uncased", base_dst)
    if not ckpt_dst.exists():
        shutil.copytree(ROOT / "checkpoints", ckpt_dst)
    return base_dst, ckpt_dst


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, math.ceil((pct / 100.0) * len(xs)) - 1))
    return xs[idx]


def normalize_probs(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, EPS, None)
    return arr / arr.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_probs(p.tolist())
    q = normalize_probs(q.tolist())
    return float(np.sum(p * np.log(p / q)))


def js(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_probs(p.tolist())
    q = normalize_probs(q.tolist())
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def divergence(plain: list[float], other: list[float]) -> dict[str, float]:
    p = normalize_probs(plain)
    q = normalize_probs(other)
    diff = q - p
    return {
        "kl_plain_to_other": kl(p, q),
        "js": js(p, q),
        "l1": float(np.abs(diff).sum()),
        "l2": float(np.linalg.norm(diff)),
        "max_abs": float(np.abs(diff).max()),
    }


def load_samples(limit: int) -> list[dict]:
    samples = DEFAULT_SAMPLES[:limit]
    if limit <= len(samples):
        return samples
    return (DEFAULT_SAMPLES * math.ceil(limit / len(DEFAULT_SAMPLES)))[:limit]


def run_plaintext_host(samples: list[dict], max_seq_len: int) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(ROOT))
    from transformer.bert_weight_loader import load_classification_model, load_tokenizer
    from transformer.plaintext_bert import classify_plaintext_hf

    model = load_classification_model("cpu")
    tokenizer = load_tokenizer()
    rows = []
    for idx, sample in enumerate(samples):
        t0 = time.perf_counter()
        label, probs = classify_plaintext_hf(model, tokenizer, sample["text"], "cpu", max_seq_len)
        elapsed = time.perf_counter() - t0
        pred = LABELS.index(label)
        rows.append(
            {
                "sample_id": idx,
                "text": sample["text"],
                "gold_label": sample.get("label"),
                "prediction": pred,
                "label": label,
                "probabilities": [float(x) for x in probs],
                "latency_s": elapsed,
            }
        )
    return rows, summarize_predictions("plaintext_host", rows, rows)


def summarize_predictions(mode: str, rows: list[dict], plain_rows: list[dict]) -> dict:
    latencies = [float(r["latency_s"]) for r in rows]
    labels = [r.get("gold_label") for r in rows if r.get("gold_label") is not None]
    correct = [
        int(r["prediction"]) == int(r["gold_label"])
        for r in rows
        if r.get("gold_label") is not None
    ]
    top1_matches = [
        int(r["prediction"]) == int(p["prediction"])
        for r, p in zip(rows, plain_rows)
    ]
    divs = [divergence(p["probabilities"], r["probabilities"]) for r, p in zip(rows, plain_rows)]
    return {
        "mode": mode,
        "status": "ok",
        "n_samples": len(rows),
        "accuracy": statistics.mean(correct) if labels else "",
        "avg_latency_s": statistics.mean(latencies) if latencies else "",
        "median_latency_s": statistics.median(latencies) if latencies else "",
        "p95_latency_s": percentile(latencies, 95) if len(latencies) >= 5 else "",
        "top1_match_with_plain": statistics.mean(top1_matches) if top1_matches else "",
        "mean_kl_plain_to_output": statistics.mean(d["kl_plain_to_other"] for d in divs) if divs else "",
        "mean_js": statistics.mean(d["js"] for d in divs) if divs else "",
        "mean_l1": statistics.mean(d["l1"] for d in divs) if divs else "",
        "mean_l2": statistics.mean(d["l2"] for d in divs) if divs else "",
        "mean_max_abs": statistics.mean(d["max_abs"] for d in divs) if divs else "",
    }


def run_crypten_docker(
    env: dict[str, str],
    bind_host: Path,
    case_dir: str,
    logs_dir: Path,
    repeat: int,
    max_seq_len: int,
    max_layers: int,
    input_json: str,
    skip_build: bool,
    bert_base_host: Path | None,
    checkpoints_host: Path | None,
) -> tuple[dict, dict]:
    project = f"psbert{datetime.now().strftime('%H%M%S')}"
    cenv = env.copy()
    cenv.update(
        {
            "PHANTOM_OUT_HOST": str(bind_host),
            "CRYPTEN_OP": "bert_full",
            "CRYPTEN_REPEAT": str(repeat),
            "CRYPTEN_MAX_SEQ_LEN": str(max_seq_len),
            "CRYPTEN_MAX_LAYERS": str(max_layers),
            "CRYPTEN_BERT_NONLINEAR": env.get("CRYPTEN_BERT_NONLINEAR", "native"),
            "CRYPTEN_INPUT_JSON": input_json,
            "CRYPTEN_OUT_DIR": f"/workspace/out/{case_dir}",
            "CRYPTEN_START_DELAY": "1",
        }
    )
    if bert_base_host and checkpoints_host:
        cenv["PHANTOM_BERT_BASE_HOST"] = str(bert_base_host)
        cenv["PHANTOM_CHECKPOINTS_HOST"] = str(checkpoints_host)
    cleanup(project, cenv)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not skip_build:
        print("[bert-docker] building CrypTen image")
        run([*compose_base(project), "build", "crypten-r0"], cenv, timeout=1800)
    r1 = subprocess.CompletedProcess([], 1, "", "")
    r0_logs = ""
    try:
        run([*compose_base(project), "up", "-d", "crypten-r0"], cenv, timeout=300)
        time.sleep(1.0)
        r1 = run([*compose_base(project), "run", "--rm", "crypten-r1"], cenv, timeout=1800, check=False)
        r0_logs = run([*compose_base(project), "logs", "--no-color", "crypten-r0"], cenv, timeout=120, check=False).stdout
    finally:
        cleanup(project, cenv)

    (logs_dir / "rank0.log").write_text(r0_logs, encoding="utf-8")
    (logs_dir / "rank1.log").write_text(r1.stdout + r1.stderr, encoding="utf-8")
    if r1.returncode != 0:
        raise RuntimeError(f"CrypTen rank1 failed:\n{r1.stdout}\n{r1.stderr}")
    rank0 = json.loads((bind_host / case_dir / "rank0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((bind_host / case_dir / "rank1.json").read_text(encoding="utf-8"))
    return rank0, rank1


def prediction_rows_from_rank(payload: dict) -> list[dict]:
    return [
        {
            "sample_id": p["sample_id"],
            "text": p["text"],
            "gold_label": p.get("gold_label"),
            "prediction": int(p["prediction"]),
            "label": p["label"],
            "probabilities": [float(x) for x in p["probabilities"]],
            "latency_s": float(p["latency_s"]),
        }
        for p in payload.get("predictions", [])
    ]


def fmt(value) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_technical_note(out: Path, summary: list[dict], max_seq_len: int, max_layers: int) -> None:
    text = [
        "# Goal 2 BERT Docker Test Technical Record",
        "",
        "## Scope",
        "",
        "- Started Goal 2 end-to-end BERT testing with timestamped CSV/JSON output.",
        "- Plaintext is measured on the host with the local SST-2 checkpoint.",
        "- CrypTen is measured in two Docker ranks over Gloo/TCP using `CRYPTEN_OP=bert_full`.",
        "- MCU Docker full BERT is not reported as a latency result yet because the current MCU Docker stack exposes accepted operator protocols, not a complete persistent 12-layer BERT role loop.",
        "",
        "## Configuration",
        "",
        f"- max_seq_len: `{max_seq_len}`",
        f"- max_layers: `{max_layers}`",
        "- model: `checkpoints/bert-sst2` when present, otherwise `bert-base-uncased`",
        "",
        "## Security Boundary",
        "",
        "- CrypTen Docker BERT currently proves the two-rank execution path, but it is still an engineering baseline.",
        "- Tokenization, embedding lookup, parts of LayerNorm, attention probability approximation, pooler, and model loading remain plaintext or shared across ranks.",
        "- MCU full secure Docker BERT requires persistent p0/p1/hp layer execution before it can be compared as a true full-inference MPC system.",
        "",
        "## Summary",
        "",
    ]
    for row in summary:
        text.append(
            f"- {row['mode']}: status={row['status']}, n={row.get('n_samples', '')}, "
            f"avg_latency_s={row.get('avg_latency_s', '')}, accuracy={row.get('accuracy', '')}"
        )
    (out / "TECHNICAL_RECORD.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--crypten-nonlinear", choices=["native", "legacy_two_quad"], default="native")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--stage-models", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "experiments" / f"{timestamp}_docker_bert_full"
    out_host = out / "shared"
    bind_host = docker_bind_root(out_host, timestamp)
    bert_base_host, checkpoints_host = docker_model_roots(args.stage_models)
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)
    out_host.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.samples)
    sample_payload = {"samples": samples}
    (out / "samples.json").write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")
    (bind_host / "samples.json").write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")

    print(f"[bert-docker] output={out}")
    plain_rows, plain_summary = run_plaintext_host(samples, args.max_seq_len)
    rank0, rank1 = run_crypten_docker(
        {**os.environ.copy(), "CRYPTEN_BERT_NONLINEAR": args.crypten_nonlinear},
        bind_host,
        "crypten_bert_full",
        logs / "crypten",
        args.repeat,
        args.max_seq_len,
        args.max_layers,
        "/workspace/out/samples.json",
        args.skip_build,
        bert_base_host,
        checkpoints_host,
    )
    crypten_rows = prediction_rows_from_rank(rank0)
    crypten_summary = summarize_predictions("crypten_docker", crypten_rows, plain_rows)
    crypten_summary["rank0_median_s"] = rank0.get("median_s", "")
    crypten_summary["rank1_median_s"] = rank1.get("median_s", "")
    crypten_summary["repeat"] = args.repeat
    crypten_summary["nonlinear"] = args.crypten_nonlinear

    mcu_summary = {
        "mode": "mcu_docker",
        "status": "blocked",
        "n_samples": "",
        "accuracy": "",
        "avg_latency_s": "",
        "median_latency_s": "",
        "p95_latency_s": "",
        "top1_match_with_plain": "",
        "mean_kl_plain_to_output": "",
        "mean_js": "",
        "mean_l1": "",
        "mean_l2": "",
        "mean_max_abs": "",
        "blocker": "MCU Docker currently has operator protocols but no persistent full 12-layer BERT p0/p1/hp inference loop.",
    }

    per_sample_rows = []
    for plain, crypten_row in zip(plain_rows, crypten_rows):
        div = divergence(plain["probabilities"], crypten_row["probabilities"])
        per_sample_rows.append(
            {
                "sample_id": plain["sample_id"],
                "text": plain["text"],
                "gold_label": plain.get("gold_label"),
                "plain_pred": plain["prediction"],
                "crypten_pred": crypten_row["prediction"],
                "plain_prob_negative": f"{plain['probabilities'][0]:.12f}",
                "plain_prob_positive": f"{plain['probabilities'][1]:.12f}",
                "crypten_prob_negative": f"{crypten_row['probabilities'][0]:.12f}",
                "crypten_prob_positive": f"{crypten_row['probabilities'][1]:.12f}",
                "plain_latency_s": f"{plain['latency_s']:.9f}",
                "crypten_latency_s": f"{crypten_row['latency_s']:.9f}",
                "crypten_top1_matches_plain": int(plain["prediction"] == crypten_row["prediction"]),
                "crypten_kl_plain_to_output": f"{div['kl_plain_to_other']:.12e}",
                "crypten_js": f"{div['js']:.12e}",
                "crypten_l1": f"{div['l1']:.12e}",
                "crypten_l2": f"{div['l2']:.12e}",
                "crypten_max_abs": f"{div['max_abs']:.12e}",
            }
        )

    summary = [plain_summary, crypten_summary, mcu_summary]
    summary_rows = [{k: fmt(v) for k, v in row.items()} for row in summary]
    write_csv(out / "bert_full_per_sample.csv", per_sample_rows)
    write_csv(out / "summary.csv", summary_rows)
    (out / "results.json").write_text(
        json.dumps(
            {
                "config": {
                    "samples": args.samples,
                    "repeat": args.repeat,
                    "max_seq_len": args.max_seq_len,
                    "max_layers": args.max_layers,
                    "crypten_nonlinear": args.crypten_nonlinear,
                    "output": str(out),
                },
                "plaintext_host": plain_rows,
                "crypten_rank0": rank0,
                "crypten_rank1": rank1,
                "mcu_docker": mcu_summary,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_technical_note(out, summary_rows, args.max_seq_len, args.max_layers)
    if bind_host.resolve() != out_host.resolve():
        if out_host.exists():
            shutil.rmtree(out_host)
        shutil.copytree(bind_host, out_host)
    print(f"[bert-docker] wrote {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
