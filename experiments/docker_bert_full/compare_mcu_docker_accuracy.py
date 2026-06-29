from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformer.bert_weight_loader import load_classification_model, load_tokenizer  # noqa: E402
from transformer.plaintext_bert import classify_plaintext_hf  # noqa: E402


LABELS = ["negative", "positive"]
EPS = 1e-12


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, EPS, None)
    return arr / arr.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize(p.tolist())
    q = normalize(q.tolist())
    return float(np.sum(p * np.log(p / q)))


def js(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize(p.tolist())
    q = normalize(q.tolist())
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def divergence(plain: list[float], other: list[float]) -> dict[str, float]:
    p = normalize(plain)
    q = normalize(other)
    diff = q - p
    return {
        "kl_plain_to_mcu": kl(p, q),
        "js": js(p, q),
        "l1": float(np.abs(diff).sum()),
        "l2": float(np.linalg.norm(diff)),
        "max_abs": float(np.abs(diff).max()),
    }


def load_samples(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    samples: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            samples.append({"text": item["text"], "label": item.get("label")})
        else:
            samples.append({"text": str(item), "label": None})
    return samples


def load_mcu_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("result", data)
    probs = [float(x) for x in result.get("probabilities", [])]
    logits = [float(x) for x in result.get("logits", [])]
    if len(probs) % 2 != 0:
        raise ValueError(f"MCU probabilities length is not divisible by 2: {len(probs)}")
    return {
        "mode": data.get("mode", "mcu_bert_session"),
        "status": data.get("status", result.get("status", "")),
        "total_s": float(result.get("total_s", data.get("critical_role_total_s", 0.0)) or 0.0),
        "probabilities": probs,
        "logits": logits,
        "predictions": result.get("predictions") or [
            int(np.argmax(probs[i : i + 2])) for i in range(0, len(probs), 2)
        ],
    }


def summarize(per_sample: list[dict], total_s: float) -> dict:
    labelled = [row for row in per_sample if row.get("gold_label") != ""]
    correct_plain = [int(row["plain_prediction"]) == int(row["gold_label"]) for row in labelled]
    correct_mcu = [int(row["mcu_prediction"]) == int(row["gold_label"]) for row in labelled]
    top1 = [int(row["plain_prediction"]) == int(row["mcu_prediction"]) for row in per_sample]
    keys = ["kl_plain_to_mcu", "js", "l1", "l2", "max_abs"]
    return {
        "mode": "mcu_vs_plaintext",
        "status": "ok",
        "n_samples": len(per_sample),
        "plain_accuracy": statistics.mean(correct_plain) if correct_plain else "",
        "mcu_accuracy": statistics.mean(correct_mcu) if correct_mcu else "",
        "top1_match_with_plain": statistics.mean(top1) if top1 else "",
        "mcu_total_s": total_s,
        "mcu_avg_latency_s": total_s / len(per_sample) if per_sample else "",
        **{f"mean_{key}": statistics.mean(float(row[key]) for row in per_sample) for key in keys},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-json", default=str(ROOT / "experiments" / "docker_bert_full" / "sst2_10_samples.json"))
    parser.add_argument("--mcu-result", required=True)
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--out-dir")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    samples = load_samples(Path(args.samples_json))
    mcu = load_mcu_result(Path(args.mcu_result))
    if len(mcu["probabilities"]) != len(samples) * 2:
        raise ValueError(
            f"MCU result contains {len(mcu['probabilities']) // 2} rows, "
            f"but samples file contains {len(samples)} rows"
        )

    out = Path(args.out_dir) if args.out_dir else ROOT / "experiments" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mcu_bert_accuracy"
    out.mkdir(parents=True, exist_ok=True)

    model = load_classification_model(args.device)
    tokenizer = load_tokenizer()
    per_sample: list[dict] = []
    plain_latency: list[float] = []
    for idx, sample in enumerate(samples):
        t0 = time.perf_counter()
        plain_label, plain_probs = classify_plaintext_hf(
            model,
            tokenizer,
            sample["text"],
            args.device,
            args.max_seq_len,
        )
        plain_elapsed = time.perf_counter() - t0
        plain_latency.append(plain_elapsed)
        plain_pred = LABELS.index(plain_label)
        mcu_probs = mcu["probabilities"][idx * 2 : idx * 2 + 2]
        mcu_logits = mcu["logits"][idx * 2 : idx * 2 + 2] if mcu["logits"] else ["", ""]
        mcu_pred = int(np.argmax(mcu_probs))
        div = divergence(plain_probs, mcu_probs)
        per_sample.append(
            {
                "sample_id": idx,
                "text": sample["text"],
                "gold_label": sample["label"] if sample["label"] is not None else "",
                "plain_prediction": plain_pred,
                "plain_label": LABELS[plain_pred],
                "plain_prob_negative": plain_probs[0],
                "plain_prob_positive": plain_probs[1],
                "plain_latency_s": plain_elapsed,
                "mcu_prediction": mcu_pred,
                "mcu_label": LABELS[mcu_pred],
                "mcu_prob_negative": mcu_probs[0],
                "mcu_prob_positive": mcu_probs[1],
                "mcu_logit_negative": mcu_logits[0],
                "mcu_logit_positive": mcu_logits[1],
                "top1_match": int(plain_pred == mcu_pred),
                **div,
            }
        )

    summary = summarize(per_sample, float(mcu["total_s"]))
    summary["plain_avg_latency_s"] = statistics.mean(plain_latency) if plain_latency else ""
    summary["mcu_result"] = str(Path(args.mcu_result))
    summary["samples_json"] = str(Path(args.samples_json))
    summary["max_seq_len"] = args.max_seq_len

    write_csv(out / "per_sample.csv", per_sample)
    write_csv(out / "summary.csv", [summary])
    (out / "results.json").write_text(
        json.dumps({"summary": summary, "per_sample": per_sample}, indent=2),
        encoding="utf-8",
    )
    print(f"[mcu-accuracy] wrote {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
